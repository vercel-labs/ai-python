"""Deprecated import path for :mod:`ai.telemetry.span`."""

import sys
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..telemetry.span import *  # noqa: F403

sys.modules[__name__] = import_module("ai.telemetry.span")
