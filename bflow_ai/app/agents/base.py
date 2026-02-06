"""
Agentic RAG Framework - Base Agent System

Mỗi domain là 1 agent với tools riêng.
Orchestrator điều phối các agents.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generator
from enum import Enum


class AgentRole(Enum):
    """Vai trò của agent"""
    DOMAIN_SPECIALIST = "domain_specialist"  # Chuyên gia domain cụ thể
    GENERALIST = "generalist"  # Trả lời câu hỏi chung
    ORCHESTRATOR = "orchestrator"  # Điều phối các agents


@dataclass
class Tool:
    """Tool mà agent có thể sử dụng"""
    name: str
    description: str
    func: callable

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@dataclass
class AgentResult:
    """Kết quả trả về từ agent"""
    agent_name: str
    content: str
    confidence: float = 1.0  # 0.0 - 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    needs_collaboration: bool = False  # Cần agent khác hỗ trợ không?

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "sources": self.sources,
            "needs_collaboration": self.needs_collaboration
        }


@dataclass
class AgentContext:
    """Context truyền giữa các agents"""
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # User ID cho phân quyền
    chat_type: str = "thinking"
    item_group: Optional[str] = None  # Chỉ PostingEngineAgent cần
    partner_group: Optional[str] = None  # Chỉ PostingEngineAgent cần
    history: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    skip_cache: bool = False  # Skip cache for this request (e.g., GENERAL_FREE)


class BaseAgent(ABC):
    """
    Base Agent class - Tất cả agents phải kế thừa từ này

    Mỗi agent:
    - Có tên và mô tả
    - Có danh sách tools
    - Có thể tự quyết định có thể xử lý query không (can_handle)
    - Execute query và trả về kết quả
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_tools()

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên agent"""
        pass

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Vai trò của agent"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Mô tả agent làm gì"""
        pass

    @abstractmethod
    def can_handle(self, context: AgentContext) -> tuple[bool, float]:
        """
        Kiểm tra agent có thể xử lý query không.
        Returns:
            (can_handle: bool, confidence: float)
        """
        pass

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """
        Thực thi query và trả về kết quả
        """
        pass

    def _register_tools(self):
        """
        Đăng ký tools cho agent.
        Override này để thêm tools.
        """
        pass

    @property
    def tools(self) -> List[Tool]:
        """Danh sách tools của agent"""
        return list(self._tools.values())

    def add_tool(self, name: str, description: str, func: callable):
        """Thêm tool cho agent"""
        self._tools[name] = Tool(name=name, description=description, func=func)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Lấy tool theo tên"""
        return self._tools.get(name)

    def stream_execute(self, context: AgentContext) -> Generator[str, None, None]:
        """
        Execute với streaming response (generator).
        Mặc định gọi execute() và yield content.
        Override này nếu cần streaming real-time.
        """
        result = self.execute(context)
        # Split content thành sentences để stream
        content = result.content

        # Simple sentence splitting cho Vietnamese
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content)

        for sentence in sentences:
            if sentence.strip():
                yield sentence + " "

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, role={self.role.value})"


class AgentOrchestrator(ABC):
    """
    Base Orchestrator - Điều phối các agents

    Orchestrator:
    - Quản lý danh sách agents
    - Route query đến agent phù hợp
    - Coordinate multi-agent collaboration
    """

    def __init__(self):
        self._agents: List[BaseAgent] = []
        self._agent_by_name: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        """Đăng ký agent"""
        self._agents.append(agent)
        self._agent_by_name[agent.name] = agent
        print(f"[Orchestrator] Registered agent: {agent.name}")

    @property
    def agents(self) -> List[BaseAgent]:
        """Danh sách tất cả agents"""
        return self._agents

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Lấy agent theo tên"""
        return self._agent_by_name.get(name)

    def find_agents_for_query(self, context: AgentContext) -> List[tuple[BaseAgent, float]]:
        """
        Tìm agents có thể xử lý query.
        Returns: List of (agent, confidence) sorted by confidence
        """
        candidates = []
        for agent in self._agents:
            can_handle, confidence = agent.can_handle(context)
            if can_handle:
                candidates.append((agent, confidence))

        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    @abstractmethod
    def route(self, context: AgentContext) -> BaseAgent:
        """
        Chọn agent phù hợp nhất để xử lý query.
        Override để implement routing logic.
        """
        pass

    @abstractmethod
    def ask(self, question: str, **kwargs) -> Generator[str, None, None]:
        """
        Main entry point - xử lý query và stream response.
        """
        pass
