"""
Vision handling for CalorAI: routes food photos to Gemini (a separate
model from the Groq text agent) and hands off a structured result.

Design decisions (for README):

1. Separate model, separate call. Images go to Gemini
   (gemini-2.5-flash, multimodal, free tier), text conversation stays
   on Groq. This satisfies the "do not run everything through one
   model" requirement and lets us pick per-modality: Groq for
   low-latency text turns, Gemini for actual image understanding
   (which Groq's text models can't do at all).

2. Structured output via Pydantic, not free-text parsing. Gemini is
   asked to return JSON matching FoodImageResult, so the handoff to
   the text agent is a typed, predictable object rather than a raw
   caption we'd have to regex out. This is also where confidence
   surfaces: `is_ambiguous` + `ambiguity_note` let the agent decide
   whether to log confidently or ask the user for confirmation,
   rather than silently guessing (an explicit green flag in the brief).

3. Caption + image resolve to ONE call, not two. If the user sends a
   photo with a caption ("half of this was my brother's"), both the
   image bytes and the caption text are sent to Gemini together in a
   single prompt, so Gemini itself reconciles them into one
   description/estimate. This avoids the failure mode where the
   image produces one "meal" and the caption produces a second,
   unrelated one downstream in the text agent.

4. Handoff shape: analyze_food_image() returns a plain string
   description (not a raw tool call) that gets fed into the SAME
   agent.run_turn() text path used for typed messages. This means the
   text agent's tool-calling logic (log_meal vs correct_last_meal,
   ambiguity handling, memory) doesn't need a separate code path for
   image-originated meals - vision's only job is turning pixels into
   a trustworthy text description handed to the one place that
   actually decides what to log.
"""

import base64
import json
import os
from typing import Optional

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

vision_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,  # low temperature: we want consistent, repeatable food identification
)


class FoodImageResult(BaseModel):
    """Structured result from the vision model, handed off to the text agent."""
    description: str = Field(description="Natural description of the food, e.g. '2 rotis with dal, roughly a full plate'")
    estimated_calories: float = Field(description="Best-effort total calorie estimate for what is shown/described")
    estimated_protein_g: float = Field(default=0.0)
    estimated_carbs_g: float = Field(default=0.0)
    estimated_fat_g: float = Field(default=0.0)
    is_ambiguous: bool = Field(description="True if the image is unclear, partially eaten, mixed with someone else's food, or otherwise hard to estimate confidently")
    ambiguity_note: Optional[str] = Field(default=None, description="If ambiguous, a short note on what's unclear - shown to the user rather than silently guessed past")


VISION_PROMPT = """You are a food identification assistant. Look at this image of food and identify what's on the plate.

{caption_context}

Return ONLY a JSON object (no markdown fences, no extra text) with this exact shape:
{{
  "description": "...",
  "estimated_calories": 000,
  "estimated_protein_g": 00,
  "estimated_carbs_g": 00,
  "estimated_fat_g": 00,
  "is_ambiguous": true/false,
  "ambiguity_note": "..." or null
}}

Rules:
- If a caption is provided, use it to correct/adjust your estimate (e.g. "half of this was my brother's" means you should roughly halve the portion you'd otherwise estimate).
- Set is_ambiguous=true if: the portion size is unclear, part of the food appears eaten already, the caption implies you're only eating part of what's shown, or you genuinely can't identify the dish confidently. Otherwise false.
- Give your best numeric estimate even when ambiguous - the ambiguity_note is what communicates the uncertainty, not a missing number.
"""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_food_image(image_path: str, caption: Optional[str] = None) -> FoodImageResult:
    """
    Send an image (and optional caption) to Gemini in a single call and
    return a structured FoodImageResult. Raises no exceptions on
    ambiguity - ambiguity is signaled via is_ambiguous/ambiguity_note
    for the caller (agent.py / main.py) to surface to the user.
    """
    image_b64 = _encode_image(image_path)
    caption_context = f'The user included this caption with the photo: "{caption}"' if caption else "No caption was provided - just the photo."

    prompt_text = VISION_PROMPT.format(caption_context=caption_context)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        ]
    )

    response = vision_llm.invoke([message])
    raw = response.content.strip()

    # Defensive cleanup: some models wrap JSON in markdown fences despite instructions.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    return FoodImageResult(**data)


def image_to_agent_message(result: FoodImageResult) -> str:
    """
    Convert a FoodImageResult into a plain-text message that can be fed
    into agent.run_turn() exactly like a typed user message. This is
    the actual "handoff" point between the vision model and the text
    agent - deliberately just a string, so the text agent's existing
    tool-calling / ambiguity-handling logic applies unchanged.
    """
    macros = (
        f"~{result.estimated_calories:.0f} kcal, "
        f"{result.estimated_protein_g:.0f}g protein, "
        f"{result.estimated_carbs_g:.0f}g carbs, "
        f"{result.estimated_fat_g:.0f}g fat"
    )
    msg = f"[Photo of food] {result.description} (vision model estimate: {macros})"
    if result.is_ambiguous:
        msg += f" NOTE: this estimate is uncertain - {result.ambiguity_note}. Confirm with the user before logging confidently, or ask what's unclear."
    return msg


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vision.py <image_path> [optional caption]")
        sys.exit(1)

    path = sys.argv[1]
    cap = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_food_image(path, cap)
    print("Raw structured result:", result.model_dump_json(indent=2))
    print("\nHandoff message to text agent:")
    print(image_to_agent_message(result))