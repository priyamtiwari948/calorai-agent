"""
CLI entry point for CalorAI Logging Agent.

Usage:
    python main.py

Input formats:
    Just type a message              -> treated as a text meal-log turn
    image: <path>                    -> photo, no caption
    image: <path> | <caption text>   -> photo with caption (e.g. "half of this was my brother's")
    stats                            -> print p50/p95 latency for text and image paths so far
    exit / quit                      -> end session

Design decisions (for README):
- Single CLI loop handles BOTH text and image turns by routing to the
  same agent.run_turn() after vision.py converts an image into a plain
  text description. This means there is exactly one place or code path
  that decides what to log/ask, regardless of input modality - required
  so the "one meal, not two" guarantee for photo+caption holds even
  when the caption itself needed no clarification.
- Latency is measured around the whole turn (image analysis + agent
  call, when applicable) since that's what the user actually waits on
  in a WhatsApp-style chat - not just the LLM call in isolation.
- p50/p95 are tracked separately for text-only turns vs image turns,
  since the assignment explicitly asks for both paths' numbers
  reported separately.
"""

import time
import statistics
import sys

from db import init_db
from agent import run_turn
from vision import analyze_food_image, image_to_agent_message


text_latencies = []
image_latencies = []
history = []  # LangGraph message history for this session


def print_stats():
    def summarize(latencies, label):
        if not latencies:
            print(f"{label}: no samples yet")
            return
        sorted_l = sorted(latencies)
        p50 = statistics.median(sorted_l)
        p95_index = min(len(sorted_l) - 1, int(round(0.95 * (len(sorted_l) - 1))))
        p95 = sorted_l[p95_index]
        print(f"{label}: n={len(sorted_l)}  p50={p50:.2f}s  p95={p95:.2f}s  min={sorted_l[0]:.2f}s  max={sorted_l[-1]:.2f}s")

    print("\n--- Latency stats ---")
    summarize(text_latencies, "Text path ")
    summarize(image_latencies, "Image path")
    print("---------------------\n")


def handle_text_turn(user_input: str):
    global history
    start = time.time()
    reply, history = run_turn(user_input, history)
    elapsed = time.time() - start
    text_latencies.append(elapsed)
    print(f"< {reply}")
    print(f"  [{elapsed:.2f}s]")


def handle_image_turn(image_path: str, caption: str | None):
    global history
    start = time.time()

    try:
        vision_result = analyze_food_image(image_path, caption)
    except FileNotFoundError:
        print(f"< Couldn't find image at '{image_path}'. Check the path and try again.")
        return
    except Exception as e:  # noqa: BLE001 - surface any vision failure clearly rather than crash the session
        print(f"< Something went wrong analyzing that image: {e}")
        return

    agent_input = image_to_agent_message(vision_result)
    # The caption, if any, is already folded into the vision analysis above -
    # we don't also re-send it as separate text, to avoid the agent treating
    # image + caption as two things to log.
    reply, history = run_turn(agent_input, history)

    elapsed = time.time() - start
    image_latencies.append(elapsed)
    print(f"< {reply}")
    print(f"  [{elapsed:.2f}s]")


def parse_image_command(line: str) -> tuple[str, str | None]:
    """Parse 'image: <path> | <caption>' or 'image: <path>' into (path, caption)."""
    body = line[len("image:"):].strip()
    if "|" in body:
        path, caption = body.split("|", 1)
        return path.strip(), caption.strip()
    return body, None


def main():
    init_db()
    print("CalorAI CLI - type a message to log a meal, ask about your totals, etc.")
    print("Commands: 'image: <path>' | 'image: <path> | <caption>' | 'stats' | 'exit'\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            print_stats()
            print("Goodbye!")
            break
        if line.lower() == "stats":
            print_stats()
            continue
        if line.lower().startswith("image:"):
            path, caption = parse_image_command(line)
            handle_image_turn(path, caption)
            continue

        handle_text_turn(line)


if __name__ == "__main__":
    main()