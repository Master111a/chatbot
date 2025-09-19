from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

class AgentInput(BaseModel):
    """
    Base class for agent input data.
    """
    query: str
    language: str = "vi"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    max_tokens: int = 8192


class AgentOutput(BaseModel):
    """
    Base class for agent output data.
    """
    response: str
    citations: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentBase(ABC):
    """
    Base class for all agents in the system.
    Defines methods that all agents must implement.
    """
    
    def __init__(self, agent_config: Dict[str, Any] = None):
        """
        Initialize the agent.
        
        Args:
            agent_config: Optional configuration for the agent
        """
        self.agent_config = agent_config or {}
        self.name = self.__class__.__name__
        
        self.stats = {
            "executions": 0,
            "avg_execution_time": 0,
            "total_execution_time": 0
        }
    
    @abstractmethod
    async def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process input query and return a response.
        Abstract method, must be implemented in child classes.
        
        Args:
            input_data: AgentInput containing query and context
            
        Returns:
            AgentOutput with response and optional metadata
        """
        pass
    