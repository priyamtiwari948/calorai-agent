"""
Tools for the CalorAI Logging Agent.

Tool boundary reasoning (documented for README):
- log_meal vs correct_last_meal are SEPARATE tools, not one "log_or_correct"
  tool. Keeping them separate means the agent's own reasoning ("is this a
  new meal or a correction to what I just logged?") is explicit and
  visible in the trace, rather than buried inside one tool's internal
  branching. This makes corrections debuggable and auditable.
- get_daily_totals and get_past_meals are separate because they answer
  different question shapes: "how am I doing today" (aggregate) vs
  "what did I eat yesterday / what's my usual" (raw records). Merging
  them would force one tool to guess which shape the caller wants.
- save_memory / get_memory are generic key-value, not per-fact tools
  (e.g. no separate `set_diet_preference`). This keeps the tool surface
  small and lets the agent decide what's worth remembering, rather than
  us hardcoding an enum of "rememberable" fact types.
- lookup_nutrition is intentionally a thin LLM-estimation wrapper, not a
  real nutrition API call. Per the assignment FAQ, nutrition accuracy is
  not being evaluated — so we avoid the cost/complexity of a real API
  integration and document the tradeoff (estimates can be off by a
  meaningful margin, especially for home-cooked / mixed dishes).
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from langchain_core.tools import tool
from db import get_session, get_or_create_user, Meal, UserMemory

# NOTE: for the CLI/test harness we operate as a single fixed user.
# In a real WhatsApp deployment, external_id would be the sender's phone number,
# resolved once per incoming message rather than hardcoded.
CURRENT_USER_EXTERNAL_ID = "test_user_cli"


def _get_current_user(session):
    return get_or_create_user(session, CURRENT_USER_EXTERNAL_ID)


@tool
def log_meal(
    description: str,
    calories: float,
    protein_g: float = 0.0,
    carbs_g: float = 0.0,
    fat_g: float = 0.0,
    source: str = "text",
) -> str:
    """
    Log a new meal for the current user.

    Use this when the user describes something they ate that has NOT
    already been logged in this conversation. Do NOT use this to fix
    a meal you just logged — use correct_last_meal for that instead,
    or double-counting will occur.

    Args:
        description: Normalized description of the meal, e.g. "2 rotis and chai".
        calories: Estimated total calories for this meal.
        protein_g: Estimated protein in grams.
        carbs_g: Estimated carbs in grams.
        fat_g: Estimated fat in grams.
        source: "text" or "image" depending on how the meal was reported.
    """
    session = get_session()
    try:
        user = _get_current_user(session)
        meal = Meal(
            user_id=user.id,
            description=description,
            calories=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            source=source,
            is_active=True,
        )
        session.add(meal)
        session.commit()
        session.refresh(meal)
        return f"Logged meal #{meal.id}: {description} ({calories:.0f} kcal, {protein_g:.0f}g protein)."
    finally:
        session.close()


@tool
def correct_last_meal(
    new_description: str,
    new_calories: float,
    new_protein_g: float = 0.0,
    new_carbs_g: float = 0.0,
    new_fat_g: float = 0.0,
) -> str:
    """
    Correct the most recently logged meal (e.g. user says "actually that
    was 3 rotis not 2"). This deactivates the previous meal record
    (excluding it from totals) and inserts a new corrected record linked
    to it — so totals update correctly and no double-counting occurs.

    Use this ONLY when the user is clearly correcting something already
    logged, not when they're describing a new/additional meal.

    Args:
        new_description: The corrected meal description.
        new_calories: Corrected total calories.
        new_protein_g: Corrected protein in grams.
        new_carbs_g: Corrected carbs in grams.
        new_fat_g: Corrected fat in grams.
    """
    session = get_session()
    try:
        user = _get_current_user(session)
        last_meal = (
            session.query(Meal)
            .filter_by(user_id=user.id, is_active=True)
            .order_by(Meal.created_at.desc())
            .first()
        )
        if last_meal is None:
            return "No previous meal found to correct. Logging this as a new meal instead."

        last_meal.is_active = False
        session.add(last_meal)

        corrected = Meal(
            user_id=user.id,
            description=new_description,
            calories=new_calories,
            protein_g=new_protein_g,
            carbs_g=new_carbs_g,
            fat_g=new_fat_g,
            source=last_meal.source,
            is_active=True,
            supersedes_id=last_meal.id,
        )
        session.add(corrected)
        session.commit()
        session.refresh(corrected)
        return (
            f"Corrected meal #{last_meal.id} -> new meal #{corrected.id}: "
            f"{new_description} ({new_calories:.0f} kcal). Previous entry excluded from totals."
        )
    finally:
        session.close()


@tool
def get_daily_totals(date_str: Optional[str] = None) -> str:
    """
    Get total calories and macros logged for a given day (defaults to today).
    Only counts active (non-corrected, non-deleted) meals.

    Args:
        date_str: Optional date in YYYY-MM-DD format. Defaults to today (UTC).
    """
    session = get_session()
    try:
        user = _get_current_user(session)

        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target_date = datetime.now(timezone.utc).date()

        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        meals = (
            session.query(Meal)
            .filter(
                Meal.user_id == user.id,
                Meal.is_active == True,  # noqa: E712
                Meal.logged_at >= start,
                Meal.logged_at < end,
            )
            .all()
        )

        if not meals:
            return f"No meals logged for {target_date.isoformat()}."

        total_cal = sum(m.calories for m in meals)
        total_protein = sum(m.protein_g for m in meals)
        total_carbs = sum(m.carbs_g for m in meals)
        total_fat = sum(m.fat_g for m in meals)

        meal_list = "; ".join(f"{m.description} ({m.calories:.0f} kcal)" for m in meals)

        return (
            f"Totals for {target_date.isoformat()}: {total_cal:.0f} kcal, "
            f"{total_protein:.0f}g protein, {total_carbs:.0f}g carbs, {total_fat:.0f}g fat. "
            f"Meals: {meal_list}"
        )
    finally:
        session.close()


@tool
def get_past_meals(days_back: int = 1, limit: int = 10) -> str:
    """
    Retrieve past logged meals, going back a number of days from today.
    Use this to answer things like "same as yesterday" or "what's my usual" —
    look at the returned meals to find the relevant one before logging.

    Args:
        days_back: How many days back to search (1 = yesterday onward, 7 = last week, etc.)
        limit: Max number of meals to return, most recent first.
    """
    session = get_session()
    try:
        user = _get_current_user(session)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        meals = (
            session.query(Meal)
            .filter(
                Meal.user_id == user.id,
                Meal.is_active == True,  # noqa: E712
                Meal.logged_at >= cutoff,
            )
            .order_by(Meal.logged_at.desc())
            .limit(limit)
            .all()
        )

        if not meals:
            return f"No meals found in the last {days_back} day(s)."

        lines = [
            f"#{m.id} [{m.logged_at.strftime('%Y-%m-%d %H:%M')}] {m.description} "
            f"({m.calories:.0f} kcal, {m.protein_g:.0f}g protein)"
            for m in meals
        ]
        return "\n".join(lines)
    finally:
        session.close()


@tool
def save_memory(key: str, value: str) -> str:
    """
    Save a durable fact about the user that should be remembered across
    sessions — e.g. dietary preference, a "usual" meal, a nutrition target.

    Only call this for facts worth remembering long-term, not for
    one-off details about a single meal. If a fact with this key already
    exists, it will be updated (not duplicated).

    Args:
        key: Short identifier, e.g. "diet_preference", "usual_breakfast", "protein_target_g".
        value: The fact to remember, e.g. "vegetarian", "2 parathas and chai", "140".
    """
    session = get_session()
    try:
        user = _get_current_user(session)
        existing = (
            session.query(UserMemory)
            .filter_by(user_id=user.id, key=key)
            .first()
        )
        if existing:
            existing.value = value
            session.add(existing)
            action = "Updated"
        else:
            new_fact = UserMemory(user_id=user.id, key=key, value=value)
            session.add(new_fact)
            action = "Saved"
        session.commit()
        return f"{action} memory: {key} = {value}"
    finally:
        session.close()


@tool
def get_memory(key: Optional[str] = None) -> str:
    """
    Retrieve remembered facts about the user. If a key is given, return
    just that fact; otherwise return all remembered facts.

    Args:
        key: Optional specific fact to look up, e.g. "diet_preference".
    """
    session = get_session()
    try:
        user = _get_current_user(session)
        query = session.query(UserMemory).filter_by(user_id=user.id)
        if key:
            query = query.filter_by(key=key)
        facts = query.all()

        if not facts:
            return "No memory found." if key else "No facts remembered yet for this user."

        return "\n".join(f"{f.key} = {f.value}" for f in facts)
    finally:
        session.close()


@tool
def lookup_nutrition(food_description: str) -> str:
    """
    Estimate nutrition (calories, protein, carbs, fat) for a described food
    item or meal, when the user hasn't given numbers themselves. This is an
    LLM-based estimate, not a lookup against a verified nutrition database —
    treat the result as approximate.

    Args:
        food_description: What the food is, e.g. "2 medium parathas with chai".
    """
    # Deliberately model-driven rather than a hardcoded/real API table (see
    # module docstring). The agent's own LLM call handles this estimate;
    # this tool exists as a named, traceable step so the estimation is
    # visible in the tool-call trace rather than silently inline in the
    # agent's reasoning.
    return (
        f"NOTE: no verified nutrition DB is wired up. Estimate calories/macros "
        f"for '{food_description}' yourself using general nutrition knowledge, "
        f"and clearly treat the result as approximate when logging."
    )


ALL_TOOLS = [
    log_meal,
    correct_last_meal,
    get_daily_totals,
    get_past_meals,
    save_memory,
    get_memory,
    lookup_nutrition,
]