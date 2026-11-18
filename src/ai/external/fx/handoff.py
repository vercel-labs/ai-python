"""Write a past into the internal fx sessions/v2 store. Verified on 0.0.17."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import time
from typing import Any

from ...types import media
from ...types import messages as messages_
from .. import _handoff_output

FORMAT_DIR = "v2"
FORMAT_VERSION = 1
PREVIEW_BYTES = 4096  # fx keeps at most this much of a result in the log line

_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def frame(payload: str) -> bytes:
    """Seal one line. `payload` is a JSON object without its closing brace."""
    raw = payload.encode()
    return raw + b',"crc":"%08x"}\n' % crc32c(raw)


class _Log:
    def __init__(self, ts: int) -> None:
        self.lines: list[bytes] = []
        self.seq = 0
        self.ts = ts

    def add(self, kind: str, fields: str = "") -> None:
        self.seq += 1
        self.ts += 1
        head = (
            f'{{"v":{FORMAT_VERSION},"seq":{self.seq},'
            f'"ts":{self.ts},"kind":"{kind}"'
        )
        self.lines.append(frame(head + fields))

    def item(
        self,
        turn: int,
        kind: str,
        data: dict[str, Any],
        blobs: list[str] | None = None,
    ) -> None:
        fields = f',"turn":{turn},"type":"{kind}","data":{_dumps(data)}'
        if blobs:
            fields += f',"blobs":{_dumps(blobs)}'
        self.add("item", fields)

    def set(self, key: str, value: Any) -> None:
        self.add("set", f',"key":"{key}","value":{_dumps(value)}')


def _preview(text: str) -> str:
    raw = text.encode()
    if len(raw) <= PREVIEW_BYTES:
        return text
    return raw[:PREVIEW_BYTES].decode(errors="ignore")


def _turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split into turns. Each turn starts with exactly one user message."""
    turns: list[list[dict[str, Any]]] = []
    for message in messages:
        if message["role"] == "user":
            if (
                turns and len(turns[-1]) == 1
            ):  # two user messages in a row: join them
                last = turns[-1][0]
                last["text"] = (last["text"] + "\n\n" + message["text"]).strip()
                last.setdefault("images", []).extend(message.get("images", []))
            else:
                turns.append(
                    [dict(message, images=list(message.get("images", [])))]
                )
        else:
            if not turns:  # the past starts with the assistant
                turns.append(
                    [
                        {
                            "role": "user",
                            "text": "(start of the earlier conversation)",
                            "images": [],
                        }
                    ]
                )
            turns[-1].append(message)
    return turns


def write_session(
    messages: list[messages_.Message],
    *,
    fx_home: str,
    cwd: str,
    prefs: dict[str, Any],
    instructions: str | None,
    session_id: str | None = None,
    now_ms: int | None = None,
) -> tuple[str, int]:
    """Write a new session. Return its id and turn count."""
    records = to_fx_records(messages)
    cwd = os.path.realpath(cwd)
    sid = (
        session_id or base64.urlsafe_b64encode(secrets.token_bytes(9)).decode()
    )
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,254}", sid):
        raise ValueError("invalid fx session id")
    root = os.path.join(fx_home, "sessions", FORMAT_DIR)
    final = os.path.join(root, sid)
    if os.path.exists(final):
        raise FileExistsError(final)
    staging = os.path.join(root, ".tmp", sid)
    os.makedirs(fx_home, mode=0o700, exist_ok=True)
    for parent in (
        os.path.join(fx_home, "sessions"),
        root,
        os.path.join(root, ".tmp"),
    ):
        try:
            os.mkdir(parent, mode=0o700)
        except FileExistsError:
            pass
    os.mkdir(staging, mode=0o700)

    try:
        os.mkdir(os.path.join(staging, "blobs"), mode=0o700)

        def put_blob(data: bytes) -> str:
            digest = hashlib.sha256(data).hexdigest()
            path = os.path.join(staging, "blobs", digest)
            if not os.path.exists(path):
                with open(path, "wb") as file:
                    file.write(data)
                os.chmod(path, 0o400)
            return digest

        log = _Log(
            now_ms if now_ms is not None else int(time.time() * 1000) - 1000
        )
        log.add(
            "session_created",
            f',"id":{_dumps(sid)},"workspace":{_dumps(cwd)},"role":"root","host":"acp"',
        )
        log.set("prefs", prefs)
        log.set(
            "permissions",
            {"schema_version": 2, "next_generation": 1, "rules": []},
        )
        log.set("language", "und")
        if instructions:
            log.set("client_prompt", instructions)

        image_id = 0
        turns = _turns(records)
        for number, turn in enumerate(turns, start=1):
            log.add("turn_started", f',"turn":{number}')
            user = turn[0]
            text, images, blobs = user["text"], [], []
            for image in user.get("images", []):
                image_id += 1
                digest = put_blob(image["data"])
                blobs.append(digest)
                text = (text + "\n" if text else "") + f"[Image #{image_id}]"
                ext = image["media_type"].split("/")[-1]
                images.append(
                    {
                        "id": image_id,
                        "path": f"seed-image-{image_id}.{ext}",
                        "media_type": image["media_type"],
                        "blob": digest,
                    }
                )
            log.item(
                number,
                "user",
                {"text": text or "(empty)", "images": images, "work_id": None},
                blobs,
            )

            open_calls: dict[str, str] = {}
            for message in turn[1:]:
                if message["role"] == "assistant":
                    calls = message.get("tool_calls", [])
                    if message.get("text") or calls:
                        log.item(
                            number,
                            "assistant",
                            {
                                "text": message.get("text", ""),
                                "provider_replay": None,
                                "standalone_response": False,
                            },
                        )
                    for call in calls:
                        open_calls[call["id"]] = call["name"]
                        log.item(
                            number,
                            "tool_call",
                            {
                                "call_id": call["id"],
                                "tool_name": call["name"],
                                "arguments_json": _dumps(
                                    call.get("arguments", {})
                                ),
                                "argument_integrity": "valid",
                                "provisional_id": None,
                                "provider_result": None,
                                "final_identity": "valid",
                                "provenance": "fx_local",
                            },
                        )
                else:
                    for result in message["results"]:
                        open_calls.pop(result["id"], None)
                        body = result["output"].encode()
                        digest = put_blob(body)
                        log.item(
                            number,
                            "tool_result",
                            {
                                "call_id": result["id"],
                                "tool_name": result["name"],
                                "status": "failure"
                                if result.get("is_error")
                                else "success",
                                "artifact_ref": f"result-{digest}.txt",
                                "tool_image_handle": None,
                                "output_bytes": len(body),
                                "stored_bytes": len(body),
                                "completeness": "complete",
                                "preview": _preview(result["output"]),
                                "provider_native": False,
                                "created_at_ms": log.ts,
                                "permission_feedback": [],
                                "committed_file_presentation": None,
                                "command_replay_ref": None,
                                "command_replay_bytes": None,
                                "command_process_presentation": None,
                                "terminal_action_presentation": None,
                            },
                            [digest],
                        )
            for (
                call_id,
                name,
            ) in (
                open_calls.items()
            ):  # a call with no result: fx refuses the turn
                body = b"(no result was recorded)"
                digest = put_blob(body)
                log.item(
                    number,
                    "tool_result",
                    {
                        "call_id": call_id,
                        "tool_name": name,
                        "status": "failure",
                        "artifact_ref": f"result-{digest}.txt",
                        "tool_image_handle": None,
                        "output_bytes": len(body),
                        "stored_bytes": len(body),
                        "completeness": "complete",
                        "preview": body.decode(),
                        "provider_native": False,
                        "created_at_ms": log.ts,
                        "permission_feedback": [],
                        "committed_file_presentation": None,
                        "command_replay_ref": None,
                        "command_replay_bytes": None,
                        "command_process_presentation": None,
                        "terminal_action_presentation": None,
                    },
                    [digest],
                )
            log.item(number, "turn_end", {"files": [], "turn_summary": None})
            log.add("turn_committed", f',"turn":{number}')

        log.add("closed")

        with open(os.path.join(staging, "lock"), "wb"):
            pass
        with open(os.path.join(staging, "log.jsonl"), "wb") as file:
            file.write(b"".join(log.lines))
            file.flush()
            os.fsync(file.fileno())
        os.chmod(os.path.join(staging, "log.jsonl"), 0o600)
        os.chmod(os.path.join(staging, "lock"), 0o600)
        os.rename(
            staging, final
        )  # atomic publish; fx never sees a half-written session
        return sid, len(turns)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def to_fx_records(messages: list[messages_.Message]) -> list[dict[str, Any]]:
    """Normalize SDK messages for the internal fx turn writer."""
    out: list[dict[str, Any]] = []
    for message in messages:
        role = "user" if message.role == "system" else message.role
        record: dict[str, Any] = {
            "role": role,
            "text": "",
            "images": [],
            "tool_calls": [],
        }

        def flush(record: dict[str, Any] = record) -> None:
            if record["text"] or record["images"] or record["tool_calls"]:
                out.append(dict(record))
                record.update(text="", images=[], tool_calls=[])

        for part in message.parts:
            if isinstance(part, messages_.TextPart):
                record["text"] += part.text
            elif (
                isinstance(part, messages_.FilePart)
                and role == "user"
                and part.media_type.startswith("image/")
            ):
                if isinstance(part.data, str) and media.is_downloadable_url(
                    part.data
                ):
                    record["text"] += f"[image: {part.data}]"
                else:
                    record["images"].append(
                        {
                            "media_type": part.media_type,
                            "data": base64.b64decode(
                                media.data_to_base64(part.data)
                            ),
                        }
                    )
            elif (
                isinstance(
                    part, messages_.ToolCallPart | messages_.BuiltinToolCallPart
                )
                and role == "assistant"
            ):
                try:
                    arguments = (
                        json.loads(part.tool_args) if part.tool_args else {}
                    )
                except json.JSONDecodeError:
                    arguments = {"raw": part.tool_args}
                record["tool_calls"].append(
                    {
                        "id": part.tool_call_id or part.id,
                        "name": part.tool_name,
                        "arguments": arguments,
                    }
                )
            elif isinstance(
                part, messages_.ToolResultPart | messages_.BuiltinToolReturnPart
            ):
                flush()
                out.append(
                    {
                        "role": "tool",
                        "results": [
                            {
                                "id": part.tool_call_id,
                                "name": part.tool_name,
                                "output": _handoff_output.format_result(part),
                                "is_error": part.result_kind == "error"
                                if isinstance(part, messages_.ToolResultPart)
                                else part.is_error,
                            }
                        ],
                    }
                )
        flush()
    return out
