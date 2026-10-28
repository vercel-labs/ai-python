"""A conversation, named as plain data.

Small enough for a database column or a workflow state blob. Carries no
callbacks: an approval hook is process-local behavior and must be supplied
again when resuming.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ...workspaces.experimental import _base


class Handle(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    """Which harness owns this conversation ("claude-code", "codex")."""
    session_id: str
    """The harness's OWN conversation id — not one we minted on the side."""
    workspace: _base.WorkspaceCoords
    options: dict[str, Any] = {}
    """Configuration the conversation was created with, so resuming
    reproduces it without a registry or an import."""
    harness_version: str | None = None
