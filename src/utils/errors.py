"""Project-specific exceptions with user-friendly messages."""
from __future__ import annotations


class FraudProjectError(Exception):
    """Base class for all errors raised deliberately by this project."""


class DatasetNotFoundError(FraudProjectError):
    """The requested dataset file does not exist."""


class EmptyDatasetError(FraudProjectError):
    """The dataset exists but contains no rows."""


class MissingColumnsError(FraudProjectError):
    """The dataset does not contain all required columns."""


class InvalidValuesError(FraudProjectError):
    """The dataset contains invalid values (strict validation only)."""


class MissingModelError(FraudProjectError):
    """A trained model file was requested but does not exist."""


class MissingDependencyError(FraudProjectError):
    """An optional or required third-party package is not installed."""


class InvalidAccountError(FraudProjectError):
    """An account ID was requested that does not exist in the data."""