"""LangGraph graph builder connecting nodes, conditional edges, and memory."""

from typing import Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.router import classify_question
from src.graph.nodes import answer_from_summary, answer_from_retrieval


def build_graph(index: Any, chunk_metadata: list[dict], summary: Any):
    """Build and compile the codebase question-answering LangGraph workflow.
    
    Args:
        index: FAISS index instance for vector retrieval.
        chunk_metadata: List of chunk metadata dicts matching index vectors.
        summary: Architecture summary dict or object.
        
    Returns:
        Compiled LangGraph instance with MemorySaver checkpointer.
    """
    builder = StateGraph(GraphState)

    def router_node(state: GraphState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"route": "specific"}
        latest_question = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        classification = classify_question(latest_question)
        return {"route": classification}

    def architecture_node(state: GraphState) -> dict:
        messages = state.get("messages", [])
        latest_question = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        answer = answer_from_summary(latest_question, summary, messages=messages)
        return {"messages": [AIMessage(content=answer)]}

    def specific_node(state: GraphState) -> dict:
        messages = state.get("messages", [])
        latest_question = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        answer = answer_from_retrieval(latest_question, index, chunk_metadata, summary, messages=messages)
        return {"messages": [AIMessage(content=answer)]}

    def route_decision(state: GraphState) -> str:
        return state.get("route", "specific")

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("architecture", architecture_node)
    builder.add_node("specific", specific_node)

    # Add edges
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "architecture": "architecture",
            "specific": "specific",
        }
    )
    builder.add_edge("architecture", END)
    builder.add_edge("specific", END)

    # Compile graph with MemorySaver checkpointer
    checkpointer = MemorySaver()
    compiled_graph = builder.compile(checkpointer=checkpointer)

    return compiled_graph
