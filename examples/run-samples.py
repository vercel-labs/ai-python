#!/usr/bin/env python3
"""Run example samples and report results.

Usage (from repo root):
    uv run examples/run-samples.py           # text-only samples
    uv run examples/run-samples.py --image   # also run image samples
    uv run examples/run-samples.py --video   # also run video samples
    uv run examples/run-samples.py --e2e     # also run e2e test scripts
    uv run examples/run-samples.py --all       # run everything
    uv run examples/run-samples.py --parallel  # run in parallel
"""

import argparse
import concurrent.futures
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "examples" / "samples"


@dataclasses.dataclass
class Sample:
    name: str
    _: dataclasses.KW_ONLY
    stdin: str | None = None
    cmd: list[str] | None = None
    extra_env: dict[str, str] | None = None
    timeout: float = 120.0


TEXT_SAMPLES = [
    Sample("stream.py"),
    Sample("stream_all.py"),
    Sample("tools_schema.py"),
    Sample("agent_simple.py"),
    Sample("agent_custom_loop.py"),
    Sample("agent_nested.py"),
    Sample("streaming_tool.py"),
    Sample("explicit_client.py"),
    Sample("middleware_simple.py"),
    Sample("multimodal_input.py"),
    Sample("check_connection.py"),
    Sample("agent_hooks.py", stdin="y\n"),
    Sample("agent_hooks_inline.py", stdin="y\n"),
    Sample("agent_hooks_serverless.py"),
    Sample("mcp_tools.py"),
    Sample("builtin_web_search.py"),
]

IMAGE_SAMPLES = [
    Sample("image_generation.py"),
    Sample("image_edit.py"),
    Sample("inline_image.py"),
]

VIDEO_SAMPLES = [
    Sample("video_generation.py"),
]

BROKEN_SAMPLES = [
    Sample("structured_output.py"),
]

# E2E tests pick non-default ports so they don't collide with a running
# dev server on 8000/5173.
_E2E_BACKEND_PORT = "18000"
_E2E_FRONTEND_PORT = "15173"

E2E_TESTS = [
    Sample(
        "multiagent-textual/test-e2e.py",
        cmd=[
            "uv",
            "run",
            "--frozen",
            "--with-editable",
            str(REPO),
            "python",
            str(REPO / "examples" / "multiagent-textual" / "test-e2e.py"),
        ],
        extra_env={"SERVER_PORT": _E2E_BACKEND_PORT},
        timeout=300.0,
    ),
    Sample(
        "fastapi-vite/e2e-test/run.sh",
        cmd=[
            "bash",
            str(REPO / "examples" / "fastapi-vite" / "e2e-test" / "run.sh"),
        ],
        extra_env={
            "BACKEND_PORT": _E2E_BACKEND_PORT,
            "FRONTEND_PORT": _E2E_FRONTEND_PORT,
        },
        timeout=300.0,
    ),
]


def _sample_cmd(sample: Sample) -> list[str]:
    if sample.cmd is not None:
        return sample.cmd
    return [
        "uv",
        "run",
        "--frozen",
        "--with-editable",
        str(REPO),
        "python",
        str(SAMPLES / sample.name),
    ]


_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def _sample_env(sample: Sample) -> dict[str, str]:
    if sample.extra_env is None:
        return _env
    return {**_env, **sample.extra_env}


def run_sample(sample: Sample) -> bool:
    print(f"{'=' * 20} {sample.name} {'=' * 20}")
    sys.stdout.flush()
    result = subprocess.run(
        _sample_cmd(sample),
        env=_sample_env(sample),
        timeout=sample.timeout,
        input=sample.stdin,
        text=True,
    )
    print()
    sys.stdout.flush()
    return result.returncode == 0


def print_summary(results: list[tuple[str, bool]]) -> bool:
    print("=" * 60)
    print("Summary:")
    any_failed = False
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            any_failed = True
    print()
    return any_failed


def run_sample_quiet(sample: Sample) -> tuple[str, bool, str]:
    try:
        result = subprocess.run(
            _sample_cmd(sample),
            env=_sample_env(sample),
            timeout=sample.timeout,
            capture_output=True,
            text=True,
            input=sample.stdin,
        )
        output = result.stdout + result.stderr
        return sample.name, result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return sample.name, False, f"TIMEOUT after {sample.timeout:g}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run example samples.")
    parser.add_argument("--text", action="store_true", help="include text samples")
    parser.add_argument("--image", action="store_true", help="include image samples")
    parser.add_argument("--video", action="store_true", help="include video samples")
    parser.add_argument("--broken", action="store_true", help="include broken samples")
    parser.add_argument("--e2e", action="store_true", help="include e2e test scripts")
    parser.add_argument("--all", action="store_true", help="run all samples")
    parser.add_argument(
        "--parallel", action="store_true", help="run samples in parallel"
    )
    args = parser.parse_args()

    has_category = args.text or args.image or args.video or args.broken or args.e2e

    samples: list[Sample] = []
    if args.text or args.all or not has_category:
        samples.extend(TEXT_SAMPLES)
    if args.image or args.all:
        samples.extend(IMAGE_SAMPLES)
    if args.video or args.all:
        samples.extend(VIDEO_SAMPLES)
    if args.broken or args.all:
        samples.extend(BROKEN_SAMPLES)
    if args.e2e or args.all:
        samples.extend(E2E_TESTS)

    results: list[tuple[str, bool]] = []

    if args.parallel:
        outputs: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {pool.submit(run_sample_quiet, s): s for s in samples}
            for future in concurrent.futures.as_completed(futures):
                name, ok, output = future.result()
                status = "PASS" if ok else "FAIL"
                print(f"  {status}  {name}")
                sys.stdout.flush()
                outputs[name] = output
                results.append((name, ok))

        passed = sorted(name for name, ok in results if ok)
        failed = sorted(name for name, ok in results if not ok)

        print()
        for name in [*passed, *failed]:
            print(f"{'=' * 20} {name} {'=' * 20}")
            if outputs[name].strip():
                print(outputs[name].rstrip())
            print()

        results.sort(key=lambda r: (not r[1], r[0]))
    else:
        for sample in samples:
            try:
                ok = run_sample(sample)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after {sample.timeout:g}s\n")
                ok = False
            results.append((sample.name, ok))

    if print_summary(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
