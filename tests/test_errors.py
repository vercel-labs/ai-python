from __future__ import annotations

import copy
import pickle

import ai


def test_unsupported_provider_error_preserves_message_and_attributes() -> None:
    """Pickle and deep copy preserve the provider ID and formatted text."""
    error = ai.UnsupportedProviderError("zzz")

    assert str(error) == "unsupported provider 'zzz'"
    for restored in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        assert restored.args == error.args
        assert restored.__dict__ == error.__dict__
        assert restored.provider_id == "zzz"
