"""Public exception types."""


class MemauditError(Exception):
    """Base error for the memaudit package."""


class MemauditConfigError(MemauditError):
    """User configuration that cannot be executed as specified."""


class MemauditPreflightError(MemauditError):
    """Fatal misconfiguration that would silently zero the audit.

    Raised from ``on_train_begin``. Raising is required: setting
    ``control.should_training_stop`` is reset by Trainer and is not reliable.
    """


class MemauditAuditError(MemauditError):
    """The audit cannot produce a valid verdict from the given artifacts."""
