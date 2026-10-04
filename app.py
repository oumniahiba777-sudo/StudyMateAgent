import os
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


st.set_page_config(
    page_title="StudyMate — LangGraph Agent",
    page_icon="🎓",
    layout="centered",
)

# ---------- Secrets ----------
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
elif os.getenv("GEMINI_API_KEY"):
    pass
else:
    st.error("GEMINI_API_KEY is not configured. Add it to Streamlit Secrets.")
    st.stop()


# ---------- Tools ----------
COURSES = {
    "python": {
        "teacher": "Dr. Ahmed",
        "credits": 4,
        "description": "Python programming and problem solving.",
    },
    "databases": {
        "teacher": "Dr. Sara",
        "credits": 3,
        "description": "Database design, SQL and data management.",
    },
    "networks": {
        "teacher": "Dr. Karim",
        "credits": 3,
        "description": "Computer networks and communication protocols.",
    },
    "artificial intelligence": {
        "teacher": "Dr. Lina",
        "credits": 4,
        "description": "Introduction to AI, agents and intelligent systems.",
    },
}


@tool
def get_course_info(course: str) -> str:
    """Get information about a university course. Use this when the student asks about a course, teacher, credits, or course description."""
    key = course.lower().strip()

    if key not in COURSES:
        matches = [name for name in COURSES if key in name or name in key]
        if len(matches) == 1:
            key = matches[0]
        else:
            return (
                "Course not found. Available courses: "
                + ", ".join(name.title() for name in COURSES)
            )

    info = COURSES[key]
    return (
        f"Course: {key.title()}\n"
        f"Teacher: {info['teacher']}\n"
        f"Credits: {info['credits']}\n"
        f"Description: {info['description']}"
    )


@tool
def make_study_plan(subject: str, days: int) -> str:
    """Create a simple study-plan recommendation for a subject over a number of days."""
    days = max(1, min(int(days), 14))
    return (
        f"Study plan for {subject} over {days} day(s):\n"
        f"- Days 1-{max(1, days//2)}: learn/review the main concepts.\n"
        f"- Next {max(1, days//3)} day(s): practice with exercises or examples.\n"
        f"- Final day(s): active recall, weak points, and a short self-test."
    )


TOOLS = [get_course_info, make_study_plan]


# ---------- LangGraph ----------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
)

model = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


def call_model(state: State):
    system = (
        "You are StudyMate, a concise university study assistant. "
        "Answer clearly and helpfully. "
        "Use the available tools whenever they contain information needed to answer. "
        "Do not invent course information. "
        "If a course is not in the tool, say so."
    )
    messages = [HumanMessage(content=system)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
builder.add_edge("tools", "agent")
graph = builder.compile()


# ---------- UI ----------
st.title("🎓 StudyMate")
st.caption("A simple LangGraph + Gemini educational agent")

with st.sidebar:
    st.subheader("About the project")
    st.write(
        "StudyMate demonstrates an AI agent that can reason over a request, "
        "call tools when needed, and return a final answer."
    )
    st.markdown("**Built with**")
    st.markdown("- LangGraph\n- LangChain\n- Gemini 2.5 Flash\n- Streamlit")
    st.markdown("**Tools**")
    st.markdown("- Course information\n- Study-plan generator")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message.type == "human":
        with st.chat_message("user"):
            st.markdown(message.content)
    elif message.type == "ai" and getattr(message, "content", ""):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input(
    "Ask about a course or request a study plan..."
)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke({"messages": st.session_state.messages})
                answer = result["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages = result["messages"]
            except Exception as e:
                st.error(
                    "The agent could not complete the request. "
                    "Check the Gemini API key and app logs."
                )
                st.caption(str(e))
