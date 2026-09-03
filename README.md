# CalorAI Logging Agent

A conversational meal-logging agent built with LangGraph. Users describe what they ate in plain, messy language — including corrections, references to past meals, and food photos — and the agent logs it, tracks running totals, and remembers durable facts across sessions.

## LangSmith Trace

https://smith.langchain.com/public/e07253ff-e916-4e05-a9b9-f855f92e36b3/r/01a06646-13be-79f3-a674-11435f2ee6c8?start_time=2026-09-03T07%3A57%3A43.742955Z

---

## Project Overview

CalorAI's product bet is that meal logging should feel like texting a friend — no forms, no dropdowns. This agent handles that end-to-end:

- Logs meals from free-text descriptions ("had 2 parathas and chai")
- Logs meals from photos, with a separate vision model
- Handles corrections without double-counting ("actually that was 3 rotis not 2")
- Resolves references like "same as yesterday" / "my usual" against real history and memory
- Remembers durable facts (diet preference, usual meals, targets) across sessions
- Decides for itself when to ask a clarifying question vs. just log a reasonable estimate

## Setup / Installation

**Requirements:** Python 3.10+, a free [Groq](https://console.groq.com) API key, a free [Google AI Studio](https://aistudio.google.com) API key (Gemini).

\`\`\`bash
git clone https://github.com/priyamtiwari948/calorai-agent.git
cd calorai-agent

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
\`\`\`

Create a `.env` file in the project root:
\`\`\`
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_gemini_key_here
\`\`\`

Initialize the database (creates `calorai.db`, a local SQLite file):
\`\`\`bash
python db.py
\`\`\`

Run the CLI:
\`\`\`bash
python main.py
\`\`\`

**CLI commands:**
| Input | Behavior |
|---|---|
| any plain text | logged/answered by the text agent |
| `image: <path>` | photo-only meal log |
| `image: <path> \| <caption>` | photo + caption, resolved as one meal |
| `stats` | print p50/p95 latency for text and image paths |
| `exit` | end session, print final stats |

> ⚠️ Only a single fixed CLI user is wired up by default (`CURRENT_USER_EXTERNAL_ID` in `agent.py`/`tools.py`). See "Multi-user support" below.

## Model Choices

| Path | Model | Why |
|---|---|---|
| Text conversation | **Groq** (`openai/gpt-oss-20b`) | Groq's inference is extremely fast, which matters directly for the WhatsApp-speed latency requirement. The model supports tool calling, which is the core requirement for this agent. Free tier, no billing needed. |
| Vision (images) | **Gemini** (`gemini-2.5-flash`) | Groq has no vision-capable free model. Gemini 2.5 Flash is multimodal, free via Google AI Studio, and gives good enough food identification for this use case. |

This satisfies the "do not run everything through one model" requirement with two genuinely different providers, chosen for different reasons (speed vs. multimodal capability) rather than arbitrarily.

**Nutrition data:** Not a real nutrition API or hardcoded table — the LLM estimates calories/macros itself (`lookup_nutrition` tool exists mainly to make this step visible/traceable, per the assignment FAQ that nutrition accuracy isn't being evaluated). Tradeoff: estimates can be meaningfully off for home-cooked or mixed dishes; documented here rather than hidden.

## How Memory Works

**What's stored:** a flat key-value table (`user_memory`: `key`, `value` per user) — e.g. `diet_preference: vegetarian`, `usual_breakfast: 2 parathas and chai`, `protein_target_g: 140`. Not a vector store — at this scale (a handful of durable facts per user) a flat table is simpler, cheaper, and easier to audit than embeddings, and just as effective.

**When it's written:** the agent calls the `save_memory` tool itself, in the same turn a fact is stated ("i'm vegetarian btw"), guided by system-prompt instructions on what counts as memory-worthy (durable facts) vs. not (one-off meal details). No separate memory-extraction pass — this trades a small chance of the model missing a worth-remembering fact for zero extra latency/LLM calls per turn.

**How it's retrieved:** *not* via the model remembering to call a tool every turn. Before every LLM call, `agent.py`'s `_load_memory_block()` pulls all of a user's facts and injects them as a short bullet list directly into the system prompt. This guarantees the facts are always in context without relying on the model's initiative, and stays cheap because there are only ever a handful of facts per user (no truncation/ranking logic needed at this scale). The `get_memory` tool still exists for the model to explicitly double-check a specific fact mid-conversation.

**What's explicitly NOT memory:** raw conversation history. Within one process run, the LangGraph message list gives short-term multi-turn context (so "actually 3 rotis" can resolve against the message just before it), but nothing here persists that transcript as long-term memory across sessions — only explicit `save_memory` calls do.

## Tool Design

| Tool | Purpose |
|---|---|
| `log_meal` | Log a brand-new meal |
| `correct_last_meal` | Correct the most recent meal — separate from `log_meal` |
| `get_daily_totals` | Aggregate calories/macros for a given day |
| `get_past_meals` | Raw recent meal records (for "same as yesterday" / "my usual") |
| `save_memory` / `get_memory` | Generic key-value durable facts |
| `lookup_nutrition` | Thin, traceable wrapper around the LLM's own nutrition estimate |

**Why `log_meal` and `correct_last_meal` are separate tools**, rather than one smarter "log_or_correct" tool: keeping them separate makes the agent's own reasoning ("is this new, or a correction to what I just logged?") an explicit, visible step in the tool-call trace, rather than hidden inside one tool's internal branching. This is directly what prevents double-counting on corrections — `correct_last_meal` deactivates (not deletes) the previous row and inserts a new one, so `get_daily_totals` (which only sums `is_active=True` rows) reflects the correction immediately.

**Why `get_daily_totals` and `get_past_meals` are separate**: they answer different question shapes — aggregate ("how am I doing today") vs. raw records ("what's my usual"). Merging them would force one tool to guess which shape the caller actually wants.

**Why `save_memory`/`get_memory` are generic key-value, not per-fact tools** (e.g. no dedicated `set_diet_preference`): keeps the tool surface small and lets the agent decide what's worth remembering, instead of us hardcoding an enum of "rememberable" fact types up front.

## Multi-turn Ambiguity Handling

Handled almost entirely through system-prompt instruction rather than separate graph nodes (see Agent Architecture below). The prompt explicitly:
- Tells the model to estimate and log rather than ask about quantities, exact slice counts, sweetness, gram weights, etc.
- Restricts image-related clarifying questions to *only* portion-sharing ambiguity flagged by the vision model's own `is_ambiguous`/`ambiguity_note` fields — not preparation details
- Allows a clarifying question only when "same as yesterday"/"my usual" has no matching memory or past meal to resolve against
- Caps this to one clarifying question per turn, and tells the model not to re-ask about something it already estimated

This went through a few iterations during testing — an earlier version over-asked (slice counts, tea sweetness) on nearly every turn; tightening the prompt to explicitly forbid those categories fixed it. See "What I'd Fix Next" for known remaining gaps.

## Agent Architecture

A single LangGraph ReAct-style loop (`agent` node ↔ `tools` node), not a multi-node graph with separate classify/extract/log stages. Rationale: for a chat agent that mainly needs to pick the right tool and call it, an explicit intent-classification node adds one more LLM call (and latency) without adding much control — the system prompt carries the decision logic, and the model's own tool choice does the routing.

## Multimodal Handling

- Images are sent to Gemini in a single call, together with any caption text, so Gemini itself reconciles them into **one** meal (e.g. "[photo] half of this was my brother's" → Gemini halves the portion estimate itself, rather than the image and caption becoming two separate downstream meals).
- Gemini returns structured JSON (via a Pydantic schema: `description`, `estimated_calories/protein/carbs/fat`, `is_ambiguous`, `ambiguity_note`) rather than free text — this is what lets ambiguity be programmatically detected and surfaced to the user instead of silently guessed past.
- The vision result is converted to a plain text string (`image_to_agent_message`) and fed into the *same* text-agent turn used for typed messages — so there is exactly one place that decides what to log, regardless of input modality.

## Latency

Measured with `time.time()` wrapped around each full turn (vision call + agent call, when applicable) in `main.py`, since that's what a WhatsApp user actually waits on.

| Path | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| Text | 3 | 6.17s | 31.34s | 2.75s | 31.34s |
| Image | 1 | 12.70s | 12.70s | 12.70s | 12.70s |

*(Small sample sizes from manual CLI testing — see note below.)*

**What we did:**
- Capped conversation history sent to the LLM to the most recent 12 messages (`MAX_HISTORY_MESSAGES` in `agent.py`). An early version resent the full growing history every turn, and per-turn latency climbed as high as 50s+ over a long session; capping it brought typical turns back under ~10s.
- Kept the memory block tiny (a handful of key-value facts) so it adds negligible prompt size regardless of session length.
- Chose Groq specifically for the text path because it's one of the fastest inference providers available for free, rather than optimizing a slower model after the fact.
- For images, both models are called only once each (one Gemini call, one Groq call) — no redundant round-trips.

**What we didn't fix / honest limitations:**
- Groq response time showed real variance turn-to-turn (2.7s to 31s) even with a small, capped context — this looks like provider-side (free-tier) queueing rather than something fixable in our code. A production deployment would want a paid tier or a fallback provider to get a tighter, more dependable p95.
- We did not implement response streaming (listed as a bonus) — with more time this would meaningfully improve *perceived* latency even where raw latency doesn't improve.
- We did not parallelize the image path's vision call against anything, since there's nothing else useful to do concurrently in a single-meal-photo flow.
- Latency numbers above come from a small number of manual CLI test turns, not a load test — with more time we'd script a repeatable batch of the assignment's test conversation set to get a larger, more statistically meaningful sample.

## Assumptions & Trade-offs

- **Single fixed CLI user** by default, rather than a real per-session login — multi-user plumbing exists in the schema (`users` table, `external_id`) but isn't wired to the CLI's input. See "What I'd fix next."
- **No hard deletes on correction** — `correct_last_meal` marks the old row `is_active=False` and inserts a new one, preserving an audit trail, rather than overwriting history.
- **Nutrition values are LLM estimates**, not verified data — acceptable per the assignment FAQ, but should not be treated as accurate for real dietary tracking.
- **SQLite over Postgres/Supabase** — chosen purely for zero-setup local persistence for this test task; a real deployment would want Postgres for concurrent-write safety.
- **Image ambiguity is not deeply calibrated** — `is_ambiguous` comes from a single Gemini call's own judgment; we did not build a separate confidence-scoring pass on top of it.

## Time Breakdown

| Phase | Time |
|---|---|
| Setup (repo, env, API keys, model troubleshooting) | ~1 hr |
| Database schema (`db.py`) | ~40 min |
| Tools (`tools.py`) | ~1 hr |
| Agent + memory (`agent.py`) | ~1 hr |
| Vision (`vision.py`) | ~50 min |
| CLI + latency measurement (`main.py`) | ~40 min |
| Debugging (over-asking, latency growth, terminal/git issues) | ~1 hr |
| README + video | ~50 min |
| **Total** | **~7 hrs** |

## What I'd Fix / Build Next With More Time

- Wire up real multi-user session isolation to the CLI (schema already supports it)
- Script the full test-conversation set as a small eval set with pass/fail criteria (a listed bonus), instead of manual CLI testing
- Add LangSmith tracing for production-grade observability into the tool-call decisions
- Investigate the Groq latency variance further (e.g. request timing breakdown, alternate model size) rather than only capping history
- Add streaming responses for better perceived latency
- Add a small set of unit tests around `correct_last_meal` and `get_daily_totals`, since those are the most correctness-sensitive tools

## Notes on AI Tool Usage

Built with Claude as a coding partner throughout — architecture discussion, writing `db.py`/`tools.py`/`agent.py`/`vision.py`/`main.py`, debugging (including a couple of real issues: empty files that were never actually saved locally, a terminal paste mishap that created garbage files, an over-asking regression in the agent's clarifying-question behavior caught via manual testing, and the latency-growth issue traced to unbounded conversation history). All code was tested locally against the assignment's test conversation set before being treated as done.

