#!/usr/bin/env python3
"""Run examples and report results.

Usage (from repo root):
    uv run examples/.test_scripts/run-examples.py  # text-only samples
    uv run examples/.test_scripts/run-examples.py --image  # also image samples
    uv run examples/.test_scripts/run-examples.py --video  # also video samples
    uv run examples/.test_scripts/run-examples.py --e2e  # also e2e test scripts
    uv run examples/.test_scripts/run-examples.py --all  # everything
    uv run examples/.test_scripts/run-examples.py --parallel  # in parallel
    uv run examples/.test_scripts/run-examples.py models/stream.py
        # run selected example files
    uv run examples/.test_scripts/run-examples.py --model MODEL
        # patch ai.get_model() to use the given model for every sample
    uv run examples/.test_scripts/run-examples.py --protocol=responses
        # patch model/provider helpers, ai.stream(), and experimental_generate()
"""

import argparse
import concurrent.futures
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO / "examples"
PATCH_SCRIPT = EXAMPLES_DIR / ".test_scripts" / "run-with-patched-model.py"


@dataclasses.dataclass
class Sample:
    name: str
    _: dataclasses.KW_ONLY
    stdin: str | None = None
    cmd: list[str] | None = None
    extra_env: dict[str, str] | None = None
    timeout: float = 120.0


TEXT_SAMPLES = [
    Sample("models/stream.py"),
    Sample("models/gateway/stream.py"),
    Sample("models/anthropic/stream.py"),
    Sample("models/openai/stream.py"),
    Sample("agents/basic.py"),
    Sample("agents/custom_loop.py"),
    Sample("agents/subagent.py"),
    Sample("agents/streaming_tool.py"),
    Sample("models/openai/openai_chat_completions.py"),
    Sample("models/openai/explicit_client.py"),
    Sample("media/multimodal_input.py"),
    Sample("models/check_connection.py"),
    Sample("agents/tool_approval.py"),
    Sample("agents/custom_hook.py"),
    Sample("agents/mcp_tools.py"),
    Sample("models/anthropic/builtin_web_search.py"),
    Sample("models/structured_output.py"),
]

IMAGE_SAMPLES = [
    Sample("media/image_generation.py"),
    Sample("media/image_edit.py"),
    Sample("media/inline_image.py"),
]

VIDEO_SAMPLES = [
    Sample("media/video_generation.py"),
]

BROKEN_SAMPLES: list[Sample] = []

# E2E tests pick non-default ports so they don't collide with a running
# dev server on 8000/5173. Each test gets its own ports so that --parallel
# doesn't make them collide with each other either.
_WEB_AGENT_BACKEND_PORT = "18001"
_WEB_AGENT_FRONTEND_PORT = "15173"

E2E_TESTS = [
    Sample(
        "apps/web_agent/e2e-test/run.sh",
        cmd=[
            "bash",
            str(
                REPO / "examples" / "apps" / "web_agent" / "e2e-test" / "run.sh"
            ),
        ],
        extra_env={
            "BACKEND_PORT": _WEB_AGENT_BACKEND_PORT,
            "FRONTEND_PORT": _WEB_AGENT_FRONTEND_PORT,
        },
        timeout=300.0,
    ),
]

KNOWN_SAMPLES = [
    *TEXT_SAMPLES,
    *IMAGE_SAMPLES,
    *VIDEO_SAMPLES,
    *BROKEN_SAMPLES,
    *E2E_TESTS,
]


def _path_key(path: Path | str) -> str:
    return Path(path).as_posix()


def _known_sample_map() -> dict[str, Sample]:
    samples: dict[str, Sample] = {}
    for sample in KNOWN_SAMPLES:
        samples[sample.name] = sample
        samples[f"examples/{sample.name}"] = sample
        samples[_path_key(EXAMPLES_DIR / sample.name)] = sample
    return samples


def _sample_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    if path.parts[:1] == ("examples",):
        return REPO / path
    return EXAMPLES_DIR / path


def _select_sample(
    name: str, known_samples: dict[str, Sample]
) -> Sample | None:
    sample = known_samples.get(name)
    if sample is not None:
        return sample
    sample = known_samples.get(_path_key(Path(name).resolve()))
    if sample is not None:
        return sample
    path = Path(name)
    if not path.is_absolute() and len(path.parts) == 1:
        matches = [s for s in KNOWN_SAMPLES if Path(s.name).name == name]
        if len(matches) == 1:
            return matches[0]
    if _sample_path(name).is_file():
        return Sample(name)
    if not path.is_absolute() and path.parts[:1] != ("examples",):
        example_path = REPO / "examples" / path
        if example_path.is_file():
            return Sample(f"examples/{_path_key(path)}")
    return None


def _sample_cmd(
    sample: Sample, model: str | None, protocol: str | None
) -> list[str]:
    if sample.cmd is not None:
        return sample.cmd
    base = [
        "uv",
        "run",
        "--frozen",
        "--group",
        "dev",
        "--with-editable",
        str(REPO),
        "python",
    ]
    if model is not None or protocol is not None:
        cmd = [*base, str(PATCH_SCRIPT)]
        if model is not None:
            cmd.extend(["--model", model])
        if protocol is not None:
            cmd.extend(["--protocol", protocol])
        return [*cmd, str(_sample_path(sample.name))]
    return [*base, str(_sample_path(sample.name))]


_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def _sample_env(sample: Sample) -> dict[str, str]:
    if sample.extra_env is None:
        return _env
    return {**_env, **sample.extra_env}


def run_sample(sample: Sample, model: str | None, protocol: str | None) -> bool:
    print(f"{'=' * 20} {sample.name} {'=' * 20}")
    sys.stdout.flush()
    result = subprocess.run(
        _sample_cmd(sample, model, protocol),
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


def run_sample_quiet(
    sample: Sample, model: str | None, protocol: str | None
) -> tuple[str, bool, str]:
    try:
        result = subprocess.run(
            _sample_cmd(sample, model, protocol),
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
    parser.add_argument(
        "--text", action="store_true", help="include text samples"
    )
    parser.add_argument(
        "--image", action="store_true", help="include image samples"
    )
    parser.add_argument(
        "--video", action="store_true", help="include video samples"
    )
    parser.add_argument(
        "--broken", action="store_true", help="include broken samples"
    )
    parser.add_argument(
        "--e2e", action="store_true", help="include e2e test scripts"
    )
    parser.add_argument("--all", action="store_true", help="run all samples")
    parser.add_argument(
        "--parallel", action="store_true", help="run samples in parallel"
    )
    parser.add_argument(
        "--model",
        help=(
            "run each sample through run-with-patched-model.py with this "
            "model id (e.g. 'openai/gpt-5.4-mini'); ignored for "
            "samples with a custom cmd"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("chat", "messages", "responses"),
        help=(
            "run each sample through run-with-patched-model.py with this "
            "underlying protocol; ignored for samples with a custom cmd"
        ),
    )
    parser.add_argument(
        "examples",
        nargs="*",
        metavar="example",
        help=(
            "example file(s) to run, e.g. models/stream.py or "
            "examples/models/stream.py"
        ),
    )
    args = parser.parse_args()

    has_category = (
        args.text or args.image or args.video or args.broken or args.e2e
    )

    samples: list[Sample] = []
    if args.examples:
        known_samples = _known_sample_map()
        for example in args.examples:
            sample = _select_sample(example, known_samples)
            if sample is None:
                parser.error(f"unknown example file: {example}")
            samples.append(sample)
    elif args.text or args.all or not has_category:
        samples.extend(TEXT_SAMPLES)
    if not args.examples and (args.image or args.all):
        samples.extend(IMAGE_SAMPLES)
    if not args.examples and (args.video or args.all):
        samples.extend(VIDEO_SAMPLES)
    if not args.examples and (args.broken or args.all):
        samples.extend(BROKEN_SAMPLES)
    if not args.examples and (args.e2e or args.all):
        samples.extend(E2E_TESTS)

    results: list[tuple[str, bool]] = []

    if args.parallel:
        outputs: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(run_sample_quiet, s, args.model, args.protocol): s
                for s in samples
            }
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
                ok = run_sample(sample, args.model, args.protocol)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after {sample.timeout:g}s\n")
                ok = False
            results.append((sample.name, ok))

    if print_summary(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
