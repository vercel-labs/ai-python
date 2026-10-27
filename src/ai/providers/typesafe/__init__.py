"""TypeSafe provider.

Usage::

    import ai
    from ai.providers.typesafe import TypeSafeProvider

    model = ai.Model(id="jev-latest", provider=TypeSafeProvider())
    result = await ai.ops.experimental.evaluate(model, state, questions)

The provider supports evaluation only. The optional upstream TypeSafe SDK is
loaded lazily when the provider creates or uses an SDK client.
"""

from .protocol import TypeSafeSystemOneProtocol
from .provider import TypeSafeProvider

__all__ = ["TypeSafeProvider", "TypeSafeSystemOneProtocol"]
