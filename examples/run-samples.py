#!/usr/bin/env python3
"""Run example samples and report results.

Usage (from repo root):
    uv run examples/run-samples.py           # text-only samples
    uv run examples/run-samples.py --image   # also run image samples
    uv run examples/run-samples.py --video   # also run video samples
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


TEXT_SAMPLES = [
    Sample("stream.py"),
    Sample("stream_all.py"),
    Sample("structured_output.py"),
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

BROKEN_SAMPLES = []


def _sample_cmd(sample: Sample) -> list[str]:
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


def run_sample(sample: Sample) -> bool:
    print(f"{'=' * 20} {sample.name} {'=' * 20}")
    sys.stdout.flush()
    result = subprocess.run(
        _sample_cmd(sample),
        env=_env,
        timeout=120,
        input=sample.stdin,
        text=True,
    )
    print()
    sys.stdout.flush()
    return result.returncode == 0


def run_sample_quiet(sample: Sample) -> tuple[str, bool, str]:
    try:
        result = subprocess.run(
            _sample_cmd(sample),
            env=_env,
            timeout=120,
            capture_output=True,
            text=True,
            input=sample.stdin,
        )
        output = result.stdout + result.stderr
        return sample.name, result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return sample.name, False, "TIMEOUT after 120s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run example samples.")
    parser.add_argument("--text", action="store_true", help="include text samples")
    parser.add_argument("--image", action="store_true", help="include image samples")
    parser.add_argument("--video", action="store_true", help="include video samples")
    parser.add_argument("--broken", action="store_true", help="include broken samples")
    parser.add_argument("--all", action="store_true", help="run all samples")
    parser.add_argument(
        "--parallel", action="store_true", help="run samples in parallel"
    )
    args = parser.parse_args()

    has_category = args.text or args.image or args.video or args.broken

    samples: list[Sample] = []
    if args.text or args.all or not has_category:
        samples.extend(TEXT_SAMPLES)
    if args.image or args.all:
        samples.extend(IMAGE_SAMPLES)
    if args.video or args.all:
        samples.extend(VIDEO_SAMPLES)
    if args.broken or args.all:
        samples.extend(BROKEN_SAMPLES)

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

        if failed:
            sys.exit(1)
    else:
        for sample in samples:
            try:
                ok = run_sample(sample)
            except subprocess.TimeoutExpired:
                print("  TIMEOUT after 120s\n")
                ok = False
            results.append((sample.name, ok))

        print("=" * 60)
        print("Summary:")
        any_failed = False
        for name, ok in results:
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {name}")
            if not ok:
                any_failed = True
        print()

        if any_failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
