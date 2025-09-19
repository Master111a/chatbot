"""
Agents module for the Agentic RAG system.
Contains all agent implementations and orchestration logic.
"""

from .agent_base import AgentBase, AgentInput, AgentOutput

def get_rag_agent():
    """Lazy import RAG Agent"""
    from .rag_agent import RAGAgent, RAGAgentInput, RAGAgentOutput
    return RAGAgent, RAGAgentInput, RAGAgentOutput

# def get_tool_agent():
#     """Lazy import for ToolAgent - DISABLED: Function Calling removed"""
#     # from .tool_agent import ToolAgent, ToolAgentInput, ToolAgentOutput
#     # return ToolAgent, ToolAgentInput, ToolAgentOutput
#     raise ImportError("ToolAgent disabled - Function Calling has been removed")

def get_chat_agent():
    """Lazy import Chat Agent"""
    from .chat_agent import ChatAgent, ChatAgentInput, ChatAgentOutput
    return ChatAgent, ChatAgentInput, ChatAgentOutput

def get_summary_agent():
    """Lazy import Summary Agent"""
    from .summary_agent import SummaryAgent, SummaryAgentInput, SummaryAgentOutput
    return SummaryAgent, SummaryAgentInput, SummaryAgentOutput

def get_reflection_agent():
    """Lazy import Reflection Agent"""
    from .reflection_agent import ReflectionAgent, ReflectionAgentInput, ReflectionAgentOutput
    return ReflectionAgent, ReflectionAgentInput, ReflectionAgentOutput

def get_hybrid_agent():
    """Lazy import Hybrid Agent"""
    from .hybrid_agent import HybridAgent
    return HybridAgent

def get_orchestrator():
    """Lazy import Orchestrator"""
    from .orchestrator import Orchestrator
    return Orchestrator

__all__ = [
    "AgentBase",
    "AgentInput", 
    "AgentOutput",
    "get_rag_agent",
    "get_chat_agent",
    "get_summary_agent",
    "get_reflection_agent",
    "get_hybrid_agent",
    "get_orchestrator"
]


AGENT_REGISTRY = {
    "RAGAgent": get_rag_agent,
    "ChatAgent": get_chat_agent,
    "SummaryAgent": get_summary_agent,
    "ReflectionAgent": get_reflection_agent,
    "HybridAgent": get_hybrid_agent
}

def get_agent_class(agent_name: str):
    """
    Dynamically get an agent class by name.
    
    Args:
        agent_name: Name of the agent class
        
    Returns:
        Agent class or None if not found
    """
    if agent_name in AGENT_REGISTRY:
        agent_classes = AGENT_REGISTRY[agent_name]()
        if isinstance(agent_classes, tuple):
            return agent_classes[0]
        else:
            return agent_classes
    return None

def list_available_agents():
    """Get list of available agent names"""
    return list(AGENT_REGISTRY.keys())