"""
Core conversational agent for CalorAI, built with LangGraph.

Design decisions (for README):

1. Single ReAct-style loop, not a multi-node graph with separate
   "classify intent" / "extract entities" / "log" nodes. For a chat
   agent that just needs to pick the right tool and call it, an
   explicit intent-classification node adds latency (one more LLM
   call) without adding much control. The system prompt carries the
   decision logic (log vs correct vs answer vs ask) and the model's
   own tool-choice does the routing. This keeps the text path to
   ideally one LLM call in the common case.

2. Memory injection happens BEFORE the LLM call, not as a tool the
   model has to remember to invoke every turn. All of the user's
   remembered facts (typically just a handful: diet preference,
   "usual" meals, targets) are fetched from user_memory and folded
   into the system prompt as a short bullet list. This guarantees
   the model always has them in context without bloating the prompt
   (facts are short, few, and deduplicated by key) and without
   relying on the model to proactively call get_memory. get_memory
   the tool still exists for when the model wants to double check
   something explicitly (e.g. "what's my usual again?").

3. save_memory is left as a tool the model calls itself, rather than
   a separate post-hoc "memory extraction" pass over every message.
   This means memory-worthy facts are captured in the same turn they
   are stated ("i'm vegetarian btw") without an extra LLM call, at
   the cost of relying on the system prompt to tell the model what's
   worth remembering. Tradeoff documented in README.

4. Conversation history is NOT memory. Within a single process run we
   keep the LangGraph message list for multi-turn context (so "my
   usual" or "actually 3 rotis" can resolve against what was just
   said), but nothing here treats raw chat transcript as long-term
   memory - only explicit save_memory calls persist across sessions.
"""

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
import os
from dotenv import load_dotenv

from tools import ALL_TOOLS
from db import get_session, get_or_create_user, UserMemory, init_db

load_dotenv()

CURRENT_USER_EXTERNAL_ID = "test_user_cli"  # see tools.py note on multi-user

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,  # low temperature: we want consistent tool-calling behavior, not creative variance
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


def _load_memory_block() -> str:
    """
    Fetch all remembered facts for the current user and render them as a
    short bullet list for the system prompt. Kept intentionally tiny -
    this is a handful of durable facts, not a document store, so no
    truncation/retrieval-ranking logic is needed at this scale.
    """
    session = get_session()
    try:
        user = get_or_create_user(session, CURRENT_USER_EXTERNAL_ID)
        facts = session.query(UserMemory).filter_by(user_id=user.id).all()
        if not facts:
            return "(no remembered facts yet)"
        return "\n".join(f"- {f.key}: {f.value}" for f in facts)
    finally:
        session.close()


SYSTEM_PROMPT_TEMPLATE = """You are CalorAI, a friendly WhatsApp-style meal logging assistant.
Users text you what they ate in casual, messy language. Your job: log it
accurately, answer questions about their intake, and remember things
worth remembering - without ever feeling like a form.

Remembered facts about this user (already saved, use them, don't re-ask):
{memory_block}

How to behave:
- If the user describes food that has NOT been logged yet, estimate its
  calories/macros (using lookup_nutrition or your own knowledge) and call
  log_meal. Don't ask for exact gram weights - reasonable estimates are fine.
- If the user is correcting something you (or they) just logged (e.g.
  "actually that was 3 rotis not 2"), call correct_last_meal - do NOT call
  log_meal again, or the meal will be double-counted.
- If the user references a past meal ("same as yesterday", "my usual"),
  call get_past_meals or check remembered facts FIRST to find out what
  that actually refers to, before logging anything. If you genuinely can't
  tell, ask a brief clarifying question rather than guessing.
- If the user asks about their intake ("how am I doing today", "how much
  protein have I had"), call get_daily_totals rather than estimating from
  memory of the conversation.
- If the user shares a durable fact about themselves (dietary preference,
  a usual meal, a nutrition target), call save_memory. Don't save one-off
  details about a single meal as memory - only things that matter beyond
  today.
- Ask a clarifying question ONLY when you genuinely can't make a reasonable
  estimate or don't know what "usual"/"same as yesterday" refers to. Do not
  ask for exact quantities, brand names, or precise macros - approximate
  and move on. Over-asking ruins the experience.
- When an image description is provided (already converted to text by the
  vision model), treat it the same as a text meal description. If the
  vision output is marked uncertain/ambiguous, surface that uncertainty to
  the user in your reply instead of silently guessing.
- Keep replies short and conversational, like a text message - not a
  report. Confirm what you logged and mention the running total only when
  relevant or asked.
"""


def build_system_message() -> SystemMessage:
    return SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(memory_block=_load_memory_block()))


def call_model(state: MessagesState):
    """Prepend a freshly-loaded system message (with current memory) and call the LLM."""
    messages = [build_system_message()] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    """Route to tools if the model requested a tool call, else end the turn."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_agent():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after tool results, let the model respond/continue

    return graph.compile()


AGENT = build_agent()


def run_turn(user_input: str, history: list) -> tuple[str, list]:
    """
    Run one conversational turn. `history` is the running list of
    LangGraph messages for this session (kept in memory by the caller,
    e.g. main.py's CLI loop) - this is short-term conversation context,
    NOT the persistent memory system.
    """
    from langchain_core.messages import HumanMessage

    history = history + [HumanMessage(content=user_input)]
    result = AGENT.invoke({"messages": history})
    new_history = result["messages"]
    reply = new_history[-1].content
    return reply, new_history


if __name__ == "__main__":
    # Quick manual smoke test.
    init_db()
    history = []
    for msg in ["i'm vegetarian btw", "had 2 parathas and chai for breakfast", "how am I doing on calories today?"]:
        print(f"\n> {msg}")
        reply, history = run_turn(msg, history)
        print(f"< {reply}")