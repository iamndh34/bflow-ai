"""
Agentic RAG Framework - Package initialization

Mỗi domain là 1 agent với tools riêng.
Orchestrator điều phối các agents.
"""
from .base import BaseAgent, AgentOrchestrator, AgentRole, AgentResult, AgentContext, Tool
from .coa_agent import COAAgent
from .posting_engine_agent import PostingEngineAgent
from .general_accounting_agent import GeneralAccountingAgent, GeneralFreeAgent
from .orchestrator import AccountingOrchestrator, get_orchestrator

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentOrchestrator",
    "AgentRole",
    "AgentResult",
    "AgentContext",
    "Tool",

    # Agents
    "COAAgent",
    "PostingEngineAgent",
    "GeneralAccountingAgent",
    "GeneralFreeAgent",

    # Orchestrator
    "AccountingOrchestrator",
    "get_orchestrator",
]
