"""Deprecated import path for :mod:`ai.telemetry`."""

import importlib
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ..telemetry import *  # noqa: F403
from ..telemetry import __all__ as __all__

if TYPE_CHECKING:
    from ..telemetry import EvaluateSpanData as EvaluateSpanData
else:
    from ..telemetry import __getattr__ as __getattr__


warnings.warn(
    "ai.experimental_telemetry is deprecated; use ai.telemetry",
    DeprecationWarning,
    stacklevel=2,
    skip_file_prefixes=(str(Path(__file__).parent.parent) + "/",),
)

# Loading the legacy submodule later would overwrite the span() re-export.
sys.modules[f"{__name__}.span"] = importlib.import_module(
    "..telemetry.span", __name__
)
