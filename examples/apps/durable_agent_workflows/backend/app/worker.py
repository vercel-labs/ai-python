from __future__ import annotations

import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)

import vercel._internal.workflow.py_sandbox  # noqa: E402

vercel._internal.workflow.py_sandbox._PASSTHROUGHS.update({"ai"})

import workflow_agent.turn  # noqa: E402, F401
