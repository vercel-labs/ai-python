# ai

## answer style

be brief, use simple language

## development

1. use `uv` to manage the project; `uv add` and `uv remove` to manage dependencies, `uv run` to run
2. after making changes run format, lint, and typecheck like ci:
    - uv run ruff format --check src tests examples
    - uv run ruff check src tests examples
    - uv run mypy
    - uv run ty check
3. to run examples that have their own `pyproject.toml`: `uv run --frozen --with-editable ~/src/py-ai/`

## code style

1. tests directory structure mirrors `src`
2. imports:
   - import by module, using the shortest unambiguous relative path. `from ..core import helpers`, `from . import streaming`. this helps improve readability and makes it easier to understand what's going on at a glance.
   - UNLESS it's `typing` — then `from typing import Foo` (there are too many of them).
   - if the module name shadows a local variable in the same file, add a trailing underscore to the import: `from ..types import messages as messages_`. do not add trailing underscores preemptively, only when there is an actual collision.
3. minimize the number of helper functions, prioritize locality of behavior.

## design principles

### 1. maximize composability

provide simple lego bricks that the user can build their feature with. each block should do one thing and be reasonably decoupled from the rest.
expose correct primitives to make it easy to modify behavior without rewriting it from scratch.

- *can the user rewrite this feature in plain python using the existing primitives?*

### 2. minimize dsl-ness and frameworkiness

express features in a way that doesn't require the user to read documentation and learn the framework. glue things together using python.
handle complexity inside the framework instead of delegating it to users.

- *does this require the user to learn a framework-specific concept that has a direct python equivalent?*

### 3. keep data model simple

ensure state is easy to serialize and deserialize, modify, and compose at any level of granularity.
move normalization and translation complexity inside the framework and keep the public data model minimal.

## references

- seal (.reference/seal): dogfooding app, ai sdk for python + workflow + ai sdk ui web app
- ai sdk (.reference/ai): vercel ai sdk for typescript
