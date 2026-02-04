"""
Templates module cho Agents

Cấu trúc:
- posting_engine: Template cho PostingEngineAgent
- coa: Template cho COAAgent
"""

from .posting_engine import get_response_template as get_posting_engine_template
from .coa import get_lookup_template, get_compare_template, get_compare_circular_template

__all__ = [
    "get_posting_engine_template",
    "get_lookup_template",
    "get_compare_template",
    "get_compare_circular_template",
]
