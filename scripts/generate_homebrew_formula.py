#!/usr/bin/env python3
"""Generate a Homebrew formula for ccgram with all Python resource blocks.

Usage: python scripts/generate_homebrew_formula.py <version>

Requires: uv (used for dependency resolution). Run from a checkout of the
released tag: dependencies come from its pyproject.toml.
For local development, prefer: brew update-python-resources alexei-led/tap/ccgram
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/{name}/{version}/json"
PYPI_PROJECT_URL = "https://pypi.org/pypi/{name}/json"
POLL_INTERVAL = 15
PYPI_API_TIMEOUT = 600
TRANSIENT_RETRIES = 3
HTTP_SERVER_ERROR = 500
# Release CI checks out the tag, so its pyproject.toml is the released one.
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

FORMULA_TEMPLATE = """\
class Ccgram < Formula
  include Language::Python::Virtualenv

  desc "Control Claude Code sessions remotely via Telegram"
  homepage "https://github.com/alexei-led/ccgram"
  url "{sdist_url}"
  sha256 "{sha256}"
  # Stated rather than left to be inferred from the sdist filename. A PyPI URL
  # is content-addressed, so without this the version is only readable by
  # decoding the tail of a hashed path, and a stale formula is hard to spot.
  version "{version}"
  license "MIT"

  depends_on "python@3.14"
  depends_on "tmux"

{resources}

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      To enable Claude Code hook notifications (done detection, interactive
      prompts, subagent tracking), run:
        ccgram hook --install

      Verify setup with:
        ccgram doctor
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/ccgram --version")
  end
end
"""


def _get_json(url: str) -> dict:
    """GET a PyPI JSON URL, retrying transient errors (5xx, network)."""
    for attempt in range(TRANSIENT_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < HTTP_SERVER_ERROR or attempt == TRANSIENT_RETRIES - 1:
                raise
        except urllib.error.URLError, TimeoutError:
            if attempt == TRANSIENT_RETRIES - 1:
                raise
        time.sleep(POLL_INTERVAL)
    raise AssertionError("unreachable")


def pypi_json(name: str, version: str) -> dict:
    """Return release metadata with the release's files under ``urls``."""
    try:
        return _get_json(PYPI_URL.format(name=name, version=version))
    except urllib.error.HTTPError as e:
        if e.code < HTTP_SERVER_ERROR:
            raise
        # The per-version endpoint can fail on its own (seen: libtmux 0.62.0
        # 503 while the project endpoint served); the project one lists files.
        files = _get_json(PYPI_PROJECT_URL.format(name=name))["releases"].get(version)
        if not files:
            raise
        return {"urls": files}


def sdist_info(name: str, version: str) -> tuple[str, str]:
    """Return (url, sha256) for a package's sdist on PyPI."""
    for f in pypi_json(name, version)["urls"]:
        if f["packagetype"] == "sdist":
            return f["url"], f["digests"]["sha256"]
    raise SystemExit(f"No sdist for {name}=={version}")


def wait_for_sdist(version: str) -> tuple[str, str]:
    """Poll PyPI until ccgram sdist is available."""
    deadline = time.monotonic() + PYPI_API_TIMEOUT
    while True:
        try:
            return sdist_info("ccgram", version)
        except urllib.error.HTTPError:
            if time.monotonic() >= deadline:
                raise
            print(f"Waiting for ccgram {version} on PyPI...", file=sys.stderr)
            time.sleep(POLL_INTERVAL)


def resolve_deps() -> list[tuple[str, str]]:
    """Resolve ccgram's runtime deps from the checked-out pyproject.toml.

    Resolving ``ccgram==<version>`` from the index instead waited on the
    index to list the fresh release, which stalled past the timeout even
    after PyPI served it (v4.11.4).
    """
    with tempfile.TemporaryDirectory() as tmp:
        reqs_out = Path(tmp) / "out.txt"
        subprocess.check_call(
            [
                "uv",
                "pip",
                "compile",
                str(PYPROJECT),
                "-o",
                str(reqs_out),
                "--no-header",
                "--no-annotate",
                "--refresh",
            ],
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
        deps = []
        for line in reqs_out.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                name, ver = line.split("==", 1)
                deps.append((name.strip(), ver.split(";")[0].strip()))
    return sorted(deps, key=lambda x: x[0].lower())


def resource_blocks(deps: list[tuple[str, str]]) -> str:
    blocks = []
    for name, ver in deps:
        url, sha = sdist_info(name, ver)
        blocks.append(
            f'  resource "{name}" do\n    url "{url}"\n    sha256 "{sha}"\n  end'
        )
    return "\n\n".join(blocks)


_EXPECTED_ARGS = 2  # script + version


def main() -> None:
    if len(sys.argv) != _EXPECTED_ARGS:
        raise SystemExit(f"Usage: {sys.argv[0]} <version>")

    version = sys.argv[1]
    print(f"Resolving ccgram {version}...", file=sys.stderr)
    sdist_url, sha256 = wait_for_sdist(version)
    deps = resolve_deps()
    print(f"Found {len(deps)} dependencies", file=sys.stderr)

    print(
        FORMULA_TEMPLATE.format(
            sdist_url=sdist_url,
            sha256=sha256,
            version=version,
            resources=resource_blocks(deps),
        )
    )


if __name__ == "__main__":
    main()
