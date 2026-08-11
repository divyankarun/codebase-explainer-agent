"""State definition for LangGraph agent workflow."""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """LangGraph state representation for the codebase explainer agent.
    
    Attributes:
        messages: Conversation history managed by add_messages reducer.
        repo_name: Identifier for the repository being analyzed.
        route: Routing classification ('architecture' or 'specific').
    """
    messages: Annotated[list, add_messages]
    repo_name: str
    route: str
