"""Compare real SDK HTTP bodies with independently pinned Gateway contracts.

See fixtures/tool-choice-contract.md for independent source revisions and
the live-evidence boundary. Only HTTP transport is mocked; request
construction is untouched.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any, Literal

import httpx2 as httpx
import pydantic
import pytest

import ai
from ai.providers.ai_gateway import GatewayV3Protocol, GatewayV4Protocol
from ai.types import messages, tools

from ..conftest import sse

FIXTURES = Path(__file__).with_name("fixtures")


# Independently transcribed from LanguageModelV3ToolChoice and
# LanguageModelV4ToolChoice at vercel/ai:
# 3bea041f28e65f371e640d2c70c302776a46387f.
# Do not derive these expectations from the Python SDK's serializer.
class ModeChoice(pydantic.BaseModel):
    type: Literal["auto", "none", "required"]


class NamedChoice(pydantic.BaseModel):
    type: Literal["tool"]
    tool_name: str = pydantic.Field(alias="toolName")


CHOICE_CONTRACTS: dict[str, pydantic.TypeAdapter[ModeChoice | NamedChoice]] = {
    "v3": pydantic.TypeAdapter(ModeChoice | NamedChoice),
    "v4": pydantic.TypeAdapter(ModeChoice | NamedChoice),
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missed mock or unexpected transport must fail, never reach a host."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unexpected real network operation")

    async def forbidden_async(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unexpected real HTTP transport")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", forbidden_async
    )


async def capture_request(
    version: str,
    choice: ai.ToolChoiceMode | ai.ToolRef | ai.ToolSelection | None,
) -> dict[str, Any]:
    context = json.loads((FIXTURES / "tool-routing-context.json").read_text())
    assert "output" not in context
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == (
            f"https://gateway.invalid/{version}/ai/language-model"
        )
        assert (
            request.headers["ai-language-model-specification-version"]
            == (version[1:])
        )
        assert request.headers["authorization"] == "Bearer offline-test-key"
        assert not captured, "unexpected second model request"
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                {
                    "type": "finish",
                    "finishReason": {"unified": "stop", "raw": "end_turn"},
                    "usage": {
                        "inputTokens": {
                            "total": 1,
                            "noCache": 1,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                        "outputTokens": {"total": 0, "text": 0, "reasoning": 0},
                    },
                }
            ),
        )

    protocol = GatewayV3Protocol() if version == "v3" else GatewayV4Protocol()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        provider = ai.get_provider(
            "gateway",
            base_url=f"https://gateway.invalid/{version}/ai",
            api_key="offline-test-key",
            client=client,
        )
        model = ai.Model(
            id="anthropic/claude-haiku-4.5",
            provider=provider,
            protocol=protocol,
        )
        params = ai.InferenceRequestParams(
            tool_calling=(
                ai.ToolCallingParams(tool_choice=choice)
                if choice is not None
                else None
            )
        )
        async with ai.stream(
            model,
            [messages.Message.model_validate(m) for m in context["messages"]],
            tools=[tools.Tool.model_validate(t) for t in context["tools"]],
            params=params,
        ) as stream:
            async for _ in stream:
                pass

    assert len(captured) == 1
    body = captured[0]
    assert "responseFormat" not in body
    assert [tool["name"] for tool in body["tools"]] == [
        "vercel_sandbox_run",
        "e2b_run_code",
        "daytona_run_code",
        "local_shell",
    ]
    for original, outgoing in zip(context["tools"], body["tools"], strict=True):
        assert outgoing["type"] == "function"
        assert outgoing["inputSchema"] == original["spec"]["params"]
    return body


@pytest.mark.parametrize("version", ["v3", "v4"])
@pytest.mark.parametrize(
    "choice,expected",
    [
        pytest.param(ai.ToolChoiceMode.AUTO, {"type": "auto"}, id="auto"),
        pytest.param(ai.ToolChoiceMode.NONE, {"type": "none"}, id="none"),
        pytest.param(
            ai.ToolChoiceMode.REQUIRED, {"type": "required"}, id="required"
        ),
        pytest.param(
            ai.ToolRef("vercel_sandbox_run"),
            {"type": "tool", "toolName": "vercel_sandbox_run"},
            id="named",
        ),
        pytest.param(None, None, id="omitted"),
    ],
)
async def test_tool_choice_request_matches_protocol(
    version: str,
    choice: ai.ToolChoiceMode | ai.ToolRef | None,
    expected: dict[str, str] | None,
) -> None:
    body = await capture_request(version, choice)
    if expected is None:
        # The optional protocol field stays absent. No default is invented.
        assert "toolChoice" not in body
    else:
        CHOICE_CONTRACTS[version].validate_python(
            body["toolChoice"], strict=True
        )
        assert body["toolChoice"] == expected


@pytest.mark.parametrize("version", ["v3", "v4"])
@pytest.mark.parametrize("mode", list(ai.ToolChoiceMode))
async def test_tool_subset_choice_request_matches_protocol(
    version: str, mode: ai.ToolChoiceMode
) -> None:
    body = await capture_request(
        version,
        ai.ToolSelection(["vercel_sandbox_run", "e2b_run_code"], mode=mode),
    )
    assert body["activeTools"] == ["e2b_run_code", "vercel_sandbox_run"]
    CHOICE_CONTRACTS[version].validate_python(body["toolChoice"], strict=True)
    assert body["toolChoice"] == {"type": mode.value}


@pytest.mark.parametrize("version", ["v3", "v4"])
@pytest.mark.parametrize("mode", ["auto", "none", "required"])
def test_legacy_scalar_choice_violates_protocol(
    version: str, mode: str
) -> None:
    with pytest.raises(pydantic.ValidationError):
        CHOICE_CONTRACTS[version].validate_python(mode, strict=True)


@pytest.mark.parametrize("version", ["v3", "v4"])
@pytest.mark.parametrize(
    "choice",
    [
        {"type": "auto"},
        {"type": "none"},
        {"type": "required"},
        {"type": "tool", "toolName": "vercel_sandbox_run"},
    ],
)
def test_object_choice_satisfies_protocol(
    version: str, choice: dict[str, str]
) -> None:
    parsed = CHOICE_CONTRACTS[version].validate_python(choice, strict=True)
    assert parsed.model_dump(by_alias=True) == choice


def test_pinned_protocol_evidence_is_intact() -> None:
    protocol = FIXTURES / "protocol"
    manifest = json.loads((protocol / "sources.json").read_text())
    assert manifest["revision"] == "3bea041f28e65f371e640d2c70c302776a46387f"
    for entry in manifest["sources"] + manifest["derived_schemas"]:
        actual = hashlib.sha256((protocol / entry["file"]).read_bytes())
        assert actual.hexdigest() == entry["sha256"], entry["file"]
    assert hashlib.sha256(
        (FIXTURES / "tool-routing-context.json").read_bytes()
    ).hexdigest() == (
        "c9777f787c0c70a5bfd19d9f4b150f1960f0586733b7f86d6a1371a6c8bb686a"
    )


def test_network_guard_rejects_unmocked_connections() -> None:
    with socket.socket() as connection:
        with pytest.raises(pytest.fail.Exception, match="unexpected real"):
            connection.connect(("127.0.0.1", 1))
