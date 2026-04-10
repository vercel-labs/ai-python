# ai

## development guidelines

1. use `uv` to manage the project; `uv add` and `uv remove` to manage dependencies, `uv run` to run
2. after making changes run format, lint, and typecheck like ci:
    - uv run ruff format --check src tests
    - uv run ruff check src tests
    - uv run mypy src tests
    - uv run pyright src tests
3. imports:
   - import by module, using the shortest unambiguous relative path. `from ..core import helpers`, `from . import streaming`
   - UNLESS it's `typing` — then `from typing import Foo` (there are too many of them).
   - if the module name shadows a local variable in the same file, add a trailing underscore to the import: `from ..types import messages as messages_`. do not add trailing underscores preemptively — only when there is an actual collision.
4. tests directory structure mirrors `src`

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

- *example*: public data model consists of a single unified `Message` type. the framework does not expose events and other intermediate steps unless the user is writing a custom adapter.

