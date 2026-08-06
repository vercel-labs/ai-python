"""AI Gateway wire protocols.

One module per protocol version (:mod:`.v3`, :mod:`.v4`), each owning its
prompt encoding, reasoning mapping, and stream-part parsing, plus
:mod:`._shared` for the version-stable pieces.  Adding a version means
adding a module; removing one means deleting it.
"""

from .v3 import GatewayV3Protocol
from .v4 import GatewayV4Protocol

__all__ = ["GatewayV3Protocol", "GatewayV4Protocol"]
