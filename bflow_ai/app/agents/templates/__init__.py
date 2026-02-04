"""
Templates module cho Agents

Cấu trúc:
- posting_engine: Template cho PostingEngineAgent
- coa: Template cho COAAgent
"""

from .posting_engine import get_response_template
from .coa import get_lookup_template, get_compare_template, get_compare_circular_template

# Alias for backward compatibility
get_posting_engine_template = get_response_template

__all__ = [
    "get_response_template",
    "get_posting_engine_template",  # Alias for backward compatibility
    "get_lookup_template",
    "get_compare_template",
    "get_compare_circular_template",
]
