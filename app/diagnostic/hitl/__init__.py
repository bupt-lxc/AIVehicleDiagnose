"""
Human-in-the-Loop (HITL) module for AutoDiagnose Multi-Agent System.
"""

from .handler import HITLHandler
from .prompts import (
    HITL_PARSE_CONFIRM,
    HITL_LOW_CONFIDENCE,
    HITL_AUDIT_CONFLICT,
    HITL_REPORT_APPROVAL,
)

__all__ = [
    "HITLHandler",
    "HITL_PARSE_CONFIRM",
    "HITL_LOW_CONFIDENCE",
    "HITL_AUDIT_CONFLICT",
    "HITL_REPORT_APPROVAL",
]
