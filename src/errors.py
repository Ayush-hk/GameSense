"""Project-specific exceptions so callers (and the dashboard) can show clear messages."""
from __future__ import annotations


class GameSenseError(Exception):
    """Base class for all expected, user-actionable GameSense errors."""


class DatasetNotFoundError(GameSenseError, FileNotFoundError):
    """The raw dataset file is not present in data/raw."""


class SchemaError(GameSenseError, ValueError):
    """The dataset does not contain the columns the pipeline needs."""


class DataReadError(GameSenseError, OSError):
    """A data file exists but cannot be read (damaged, truncated or not the expected format)."""


class TimestampParseError(GameSenseError, ValueError):
    """Timestamps could not be parsed reliably."""


class EmptyDatasetError(GameSenseError, ValueError):
    """No usable rows remain after validation."""


class InsufficientHistoryError(GameSenseError, ValueError):
    """The data span is too short for the configured observation/prediction windows."""


class LeakageError(GameSenseError, RuntimeError):
    """A feature window overlaps its label window (temporal leakage)."""


class ArtifactMissingError(GameSenseError, FileNotFoundError):
    """A generated artifact (model, processed table, report) has not been produced yet."""


class PredictionInputError(GameSenseError, ValueError):
    """User-supplied prediction input failed validation."""
