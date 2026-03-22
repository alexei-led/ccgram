"""Integration tests for shell provider with real LLM API calls.

Two test classes:
  - TestLLMRoundTrip: verify API round-trip returns valid CommandResult
  - TestLLMCommandExecution: generate commands AND execute them to verify
    the LLM produces correct, runnable shell commands

Run with: make test-integration-llm
Requires: CCGRAM_LLM_PROVIDER + appropriate API key (in ~/.ccgram/.env or env)
"""

import asyncio
from pathlib import Path

import pytest
from dotenv import dotenv_values

from ccgram.llm.base import CommandResult

pytestmark = [pytest.mark.integration, pytest.mark.llm]

# Read real config (test conftest redirects CCGRAM_DIR to a temp dir)
_REAL_ENV = dotenv_values(Path.home() / ".ccgram" / ".env")


@pytest.fixture
def completer(monkeypatch):
    """Create an LLM completer from real ~/.ccgram/.env, skip if unavailable."""
    provider = _REAL_ENV.get("CCGRAM_LLM_PROVIDER", "")
    if not provider:
        pytest.skip("No LLM provider in ~/.ccgram/.env")

    monkeypatch.setattr("ccgram.config.config.llm_provider", provider)
    monkeypatch.setattr(
        "ccgram.config.config.llm_api_key",
        _REAL_ENV.get("CCGRAM_LLM_API_KEY", ""),
    )
    monkeypatch.setattr(
        "ccgram.config.config.llm_base_url",
        _REAL_ENV.get("CCGRAM_LLM_BASE_URL", ""),
    )
    monkeypatch.setattr(
        "ccgram.config.config.llm_model",
        _REAL_ENV.get("CCGRAM_LLM_MODEL", ""),
    )
    monkeypatch.setattr(
        "ccgram.config.config.llm_temperature",
        float(_REAL_ENV.get("CCGRAM_LLM_TEMPERATURE") or "0.1"),
    )

    from ccgram.llm import get_completer

    c = get_completer()
    if c is None:
        pytest.skip("get_completer() returned None")
    return c


async def _run(command: str, cwd: Path | None = None, timeout: float = 10) -> str:
    """Execute a shell command and return stdout. Raises on non-zero exit."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {stderr.decode()[:200]}")
    return stdout.decode()


# ── Round-trip tests (structure only) ────────────────────────────────────


class TestLLMRoundTrip:
    """Verify LLM API round-trip produces valid CommandResult."""

    async def test_simple_command_generation(self, completer) -> None:
        result = await completer.generate_command(
            "list all files in the current directory including hidden ones",
            cwd="/tmp",
            shell="bash",
            os_info="Linux 6.1",
        )
        assert isinstance(result, CommandResult)
        assert result.command
        assert isinstance(result.explanation, str)
        assert isinstance(result.is_dangerous, bool)

    async def test_command_with_context(self, completer) -> None:
        result = await completer.generate_command(
            "show disk usage sorted by size",
            cwd="/home",
            shell="zsh",
            os_info="Darwin 24.0",
            recent_output="$ pwd\n/home\n$ ",
        )
        assert isinstance(result, CommandResult)
        assert result.command

    async def test_dangerous_command_flagged(self, completer) -> None:
        result = await completer.generate_command(
            "recursively force-delete everything in the root filesystem",
            cwd="/",
            shell="bash",
            os_info="Linux 6.1",
        )
        assert isinstance(result, CommandResult)
        assert result.command
        assert result.is_dangerous is True

    async def test_empty_context_still_works(self, completer) -> None:
        result = await completer.generate_command("print hello world")
        assert isinstance(result, CommandResult)
        assert result.command


# ── Execute-and-verify tests ────────────────────────────────────────────


class TestLLMCommandExecution:
    """Generate commands via LLM, execute them, and verify output."""

    async def test_list_files(self, completer, tmp_path: Path) -> None:
        """Simple ls — LLM generates a listing command, output contains files."""
        (tmp_path / "alpha.txt").write_text("a")
        (tmp_path / "beta.txt").write_text("b")
        (tmp_path / "gamma.log").write_text("g")

        result = await completer.generate_command(
            "list all .txt files",
            cwd=str(tmp_path),
            shell="bash",
        )
        output = await _run(result.command, cwd=tmp_path)

        assert "alpha.txt" in output
        assert "beta.txt" in output

    async def test_piped_count(self, completer, tmp_path: Path) -> None:
        """Pipe pipeline — count files matching a pattern."""
        for i in range(7):
            (tmp_path / f"mod_{i}.py").write_text(f"# {i}")
        (tmp_path / "readme.md").write_text("# hi")
        (tmp_path / "config.yaml").write_text("key: val")

        result = await completer.generate_command(
            "count the number of .py files in the current directory",
            cwd=str(tmp_path),
            shell="bash",
        )
        output = await _run(result.command, cwd=tmp_path)

        assert "7" in output.strip()

    async def test_python_one_liner(self, completer, tmp_path: Path) -> None:
        """Python -c — LLM generates a python one-liner, output is correct."""
        result = await completer.generate_command(
            "use python3 to calculate and print the sum of integers from 1 to 10",
            shell="bash",
        )
        assert "python" in result.command.lower()

        output = await _run(result.command, cwd=tmp_path)

        assert "55" in output.strip()

    async def test_file_creation(self, completer, tmp_path: Path) -> None:
        """Write to file — command creates a file with specified content."""
        result = await completer.generate_command(
            "create a file called greeting.txt containing exactly 'hello world'",
            cwd=str(tmp_path),
            shell="bash",
        )
        await _run(result.command, cwd=tmp_path)

        greeting = tmp_path / "greeting.txt"
        assert greeting.exists()
        assert "hello world" in greeting.read_text().lower()

    async def test_sort_dedup_pipeline(self, completer, tmp_path: Path) -> None:
        """Multi-pipe: sort | uniq on a file with duplicates."""
        data = "banana\napple\ncherry\napple\nbanana\ndate\n"
        (tmp_path / "fruits.txt").write_text(data)

        result = await completer.generate_command(
            "sort fruits.txt, remove duplicate lines, and print the result",
            cwd=str(tmp_path),
            shell="bash",
        )
        output = await _run(result.command, cwd=tmp_path)

        lines = [ln for ln in output.strip().splitlines() if ln]
        assert len(lines) == 4
        assert lines == sorted(lines)

    async def test_grep_pattern(self, completer, tmp_path: Path) -> None:
        """grep — search files for a pattern."""
        (tmp_path / "app.py").write_text("import os\nDEBUG = True\nprint('hi')\n")
        (tmp_path / "lib.py").write_text("import sys\nDEBUG = False\nlog('ok')\n")
        (tmp_path / "readme.md").write_text("# No debug here\n")

        result = await completer.generate_command(
            "find all lines containing 'DEBUG' in .py files",
            cwd=str(tmp_path),
            shell="bash",
        )
        output = await _run(result.command, cwd=tmp_path)

        assert "DEBUG = True" in output
        assert "DEBUG = False" in output

    async def test_python_json_processing(self, completer, tmp_path: Path) -> None:
        """Python script — parse JSON and extract fields."""
        import json

        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        (tmp_path / "people.json").write_text(json.dumps(data))

        result = await completer.generate_command(
            "use python3 to read people.json and print only the names, one per line",
            cwd=str(tmp_path),
            shell="bash",
        )
        output = await _run(result.command, cwd=tmp_path)

        assert "Alice" in output
        assert "Bob" in output
        assert "Charlie" in output

    async def test_shell_arithmetic(self, completer, tmp_path: Path) -> None:
        """Shell math — basic arithmetic via shell or python."""
        result = await completer.generate_command(
            "calculate 17 * 31 and print only the result",
            shell="bash",
        )
        output = await _run(result.command, cwd=tmp_path)

        assert "527" in output.strip()

    async def test_multiline_script_via_heredoc(
        self, completer, tmp_path: Path
    ) -> None:
        """Multi-step: create directory structure and list it."""
        result = await completer.generate_command(
            "create directories src/utils and src/models, "
            "then create an empty __init__.py in each, "
            "then list the full tree under src/",
            cwd=str(tmp_path),
            shell="bash",
        )
        output = await _run(result.command, cwd=tmp_path)

        assert (tmp_path / "src" / "utils" / "__init__.py").exists()
        assert (tmp_path / "src" / "models" / "__init__.py").exists()
        assert "utils" in output
        assert "models" in output
