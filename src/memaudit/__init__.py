"""memaudit - training-data memorization auditor for HF Trainer / TRL fine-tunes."""

from memaudit.audit import run_audit
from memaudit.callback import MemorizationAuditCallback
from memaudit.canaries import generate_canaries
from memaudit.constants import TOOL_VERSION
from memaudit.exceptions import (
    MemauditAuditError,
    MemauditConfigError,
    MemauditError,
    MemauditPreflightError,
)
from memaudit.injection import inject

__version__ = TOOL_VERSION

__all__ = [
    "generate_canaries",
    "inject",
    "MemorizationAuditCallback",
    "run_audit",
    "MemauditError",
    "MemauditConfigError",
    "MemauditPreflightError",
    "MemauditAuditError",
    "__version__",
]
