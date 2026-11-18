import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "legacy_import",
    [
        "legacy = ai.experimental_telemetry",
        "from ai import experimental_telemetry as legacy",
        "import ai.experimental_telemetry as legacy",
    ],
)
def test_experimental_telemetry_import_warns(legacy_import: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-Werror::DeprecationWarning",
            "-c",
            f"""
import importlib
import sys
import warnings

import ai
from ai.telemetry import DictSink

assert 'ai.experimental_telemetry' not in sys.modules
assert 'ai.telemetry.otel' not in sys.modules

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', DeprecationWarning)
    {legacy_import}

assert len(caught) == 1
assert caught[0].category is DeprecationWarning
assert 'use ai.telemetry' in str(caught[0].message)
assert caught[0].filename == '<string>'
assert legacy.DictSink is DictSink

for submodule in ('span', 'otel', 'utils.console', 'utils.viewer'):
    old = importlib.import_module(f'ai.experimental_telemetry.{{submodule}}')
    assert old is importlib.import_module(f'ai.telemetry.{{submodule}}')

assert legacy.span is ai.telemetry.span
""",
        ],
        check=True,
    )
