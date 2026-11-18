"""Deprecated import path for :mod:`ai.telemetry.utils.viewer`."""

import sys
from importlib import import_module
from typing import TYPE_CHECKING

from ...telemetry.utils.viewer import main

if TYPE_CHECKING:
    from ...telemetry.utils.viewer import *  # noqa: F403

if __name__ == "__main__":
    main()
else:
    sys.modules[__name__] = import_module("ai.telemetry.utils.viewer")
