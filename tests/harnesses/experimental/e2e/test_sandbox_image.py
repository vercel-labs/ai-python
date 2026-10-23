"""A sandbox booted from a non-default image is that image, end to end.

Each test boots a real microVM. The images are Vercel's managed ones, chosen
because their contents differ in ways a shell can see: `ubuntu` has no
Node.js, `node:24` has Node 24 and pnpm, and the default (`universal`) has
Node too. So "it booted the image asked for" is checked by what is on PATH,
not only by what the platform reports.

Neither `ubuntu` nor `node:24` has python3, which a process's stdin needs in
the VM; the default image does. Measured, so pinned here too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai.workspaces.experimental import VercelSandbox, copy
from ai.workspaces.experimental._gateway import KEY_VAR, vercel_ai_gateway
from ai.workspaces.experimental.errors import WorkspaceError
from tests.harnesses.experimental.conftest import SPECS, exact
from tests.workspaces.experimental.conftest import (
    ENV_LOCAL,
    SandboxCredentials,
    sandbox_credentials,
)

pytestmark = [pytest.mark.sandbox, pytest.mark.timeout(600)]

UBUNTU = "vercel/sandbox/ubuntu:latest"
NODE24 = "vercel/sandbox/node:24"
# Where `apt-get` in these images fetches from (/etc/apt/sources.list.d).
APT_HOSTS = ("archive.ubuntu.com", "security.ubuntu.com")

needs_gateway = pytest.mark.skipif(
    not ENV_LOCAL.get(KEY_VAR, os.environ.get(KEY_VAR)),
    reason=f"needs {KEY_VAR}",
)


def _creds() -> SandboxCredentials:
    creds = sandbox_credentials()
    if creds is None:
        pytest.skip(
            "needs VERCEL_TOKEN/TEAM_ID/PROJECT_ID or VERCEL_OIDC_TOKEN"
        )
    return creds


async def test_the_bare_ubuntu_image_has_no_node() -> None:
    """The default image ships Node, so its absence proves the image took."""
    async with VercelSandbox(**_creds(), image=UBUNTU) as ws:
        # The platform reports the digest it resolved the tag to.
        assert ws.image and ws.image.startswith("vercel/sandbox/ubuntu@sha256:")
        found = await ws.exec(["sh", "-c", "command -v node"], timeout=60)
    assert found.exit_code != 0, f"node is on PATH: {found.stdout!r}"


async def test_the_node_image_has_its_pinned_major_and_pnpm() -> None:
    async with VercelSandbox(**_creds(), image=NODE24) as ws:
        assert ws.image and ws.image.startswith("vercel/sandbox/node@sha256:")
        node = await ws.exec(["node", "--version"], timeout=60)
        pnpm = await ws.exec(["pnpm", "--version"], timeout=60)
    assert node.stdout.startswith("v24."), node.stdout
    assert pnpm.exit_code == 0, pnpm.stderr


async def test_without_an_image_the_default_still_boots() -> None:
    async with VercelSandbox(**_creds()) as ws:
        node = await ws.exec(["node", "--version"], timeout=60)
    assert node.exit_code == 0, node.stderr


async def test_the_workdir_works_in_a_custom_image(tmp_path: Path) -> None:
    """`_open` makes the workdir; a custom image must not break that, nor
    copying a tree in and reading it back."""
    src = tmp_path / "proj"
    src.mkdir()
    (src / "util.py").write_text("def add(a, b):\n    return a + b\n")
    async with VercelSandbox(**_creds(), image=UBUNTU) as ws:
        assert await copy(src, ws / ".") == 1
        assert "def add" in await ws.read_text("util.py")
        pwd = await ws.exec(["pwd"], timeout=60)
    assert pwd.stdout.strip() == ws.path


async def test_a_reconnect_reports_the_image_its_vm_booted_from() -> None:
    """afk reconnects by name and never passes image=; the image must still
    be knowable from the reconnected object."""
    async with VercelSandbox(**_creds(), image=UBUNTU, keep=True) as ws:
        name, booted = ws.name, ws.image
    assert name and booted
    try:
        async with VercelSandbox(**_creds(), name=name) as back:
            assert back.image == booted
    finally:
        async with VercelSandbox(**_creds(), name=name, keep=False):
            pass


async def test_an_image_that_does_not_exist_is_an_error_not_a_hang() -> None:
    from vercel.sandbox import SandboxApiError

    with pytest.raises(SandboxApiError, match="Image not found"):
        async with VercelSandbox(
            **_creds(), image="ai-python-no-such-image:never"
        ):
            pass


async def test_without_python3_writing_to_a_process_says_so() -> None:
    """Measured: the first write failed as a bare ClosedResourceError, and a
    harness reported only "could not write to the agent"."""
    async with VercelSandbox(**_creds(), image=UBUNTU) as ws:
        proc = await ws.spawn(["cat"])
        with pytest.raises(WorkspaceError, match="python3"):
            await proc.write("hello\n")


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.timeout(900)
@needs_gateway
async def test_a_harness_runs_in_a_custom_image(harness_kind: str) -> None:
    """The point of the feature: an agent works in the image it was given.

    node:24 ships npm but neither python3 nor a harness CLI. Given python3,
    as any image meant for harnesses must, the adapter installs its CLI
    itself: a path the default image, which has both CLIs, never takes."""
    async with VercelSandbox(
        **_creds(),
        image=NODE24,
        gateway=vercel_ai_gateway(),
        allow_hosts=APT_HOSTS,
        execution_time_limit=900,
    ) as ws:
        # The gateway's firewall passes allowed hosts over HTTPS only.
        # Measured: apt's default http:// mirrors had the connection reset.
        apt = await ws.exec(
            [
                "sh",
                "-c",
                "sudo sed -i 's|http://|https://|'"
                " /etc/apt/sources.list.d/ubuntu.sources"
                " && sudo apt-get update -qq"
                " && sudo DEBIAN_FRONTEND=noninteractive"
                " apt-get install -y -qq python3-minimal",
            ],
            timeout=300,
        )
        assert apt.exit_code == 0, apt.stderr
        await ws.write_text("util.py", "def add(a, b):\n    return a + b\n")
        async with SPECS[harness_kind](workspace=ws) as harness:
            result = await harness.run(
                "Read util.py and reply with the function name only."
            )
    exact(result.text, "add")
