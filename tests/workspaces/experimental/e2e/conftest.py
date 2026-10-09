"""Workspace tests that put an agent in the workspace.

They need what the harness suite has: `harness_kind`, and an `any_workspace`
that can host a harness (credentials and config provisioned in a sandbox).
"""

from tests.harnesses.experimental.conftest import (
    any_workspace as any_workspace,
)
from tests.harnesses.experimental.conftest import (
    harness_kind as harness_kind,
)
