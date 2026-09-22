"""Tests for the host-side admin channel (TASK-128)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ccgram.admin as admin_mod
from ccgram.admin import (
    append_admin_result,
    execute_admin_command,
    read_new_commands,
    submit_admin_command,
    wait_for_result,
)
from ccgram.session import SessionManager
from ccgram.thread_router import thread_router
from ccgram.window_state_store import window_store


@pytest.fixture(autouse=True)
def _admin_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    thread_router.reset()
    window_store.window_states.clear()
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    SessionManager()
    yield
    thread_router.reset()


_RECORD_SEQ = iter(range(1000))


def _record(command: str, **args) -> dict:
    return {"id": f"cmd-{next(_RECORD_SEQ)}", "command": command, "args": args}


class TestSubmitAndRead:
    def test_submit_appends_and_read_returns_once(self, tmp_path) -> None:
        command_id = submit_admin_command(
            "bind", user_id=1, chat_id=42, thread_id=100, window_id="@0"
        )
        path = tmp_path / "admin_commands.jsonl"
        assert path.exists()

        records, offset = read_new_commands(path, 0)
        assert [r["id"] for r in records] == [command_id]
        assert records[0]["command"] == "bind"

        # No new content: second read returns nothing and keeps the offset.
        again, offset2 = read_new_commands(path, offset)
        assert again == []
        assert offset2 == offset

    def test_unknown_command_rejected(self) -> None:
        with pytest.raises(ValueError):
            submit_admin_command("explode")

    def test_malformed_lines_skipped(self, tmp_path) -> None:
        path = tmp_path / "admin_commands.jsonl"
        path.write_text("not json\n" + json.dumps(_record("sync")) + "\n")
        records, _ = read_new_commands(path, 0)
        assert [r["command"] for r in records] == ["sync"]


class TestBind:
    async def test_bind_live_window(self) -> None:
        with (
            patch(
                "ccgram.multiplexer.reconciliation.window_presence",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("ccgram.multiplexer.multiplexer"),
        ):
            result = await execute_admin_command(
                _record(
                    "bind",
                    user_id=1,
                    chat_id=42,
                    thread_id=100,
                    window_id="@7",
                ),
                client=MagicMock(),
            )
        assert result["ok"], result["detail"]
        assert thread_router.get_window_for_chat_thread(42, 100) == "@7"

    async def test_bind_dead_window_fails_closed(self) -> None:
        with (
            patch(
                "ccgram.multiplexer.reconciliation.window_presence",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("ccgram.multiplexer.multiplexer"),
        ):
            result = await execute_admin_command(
                _record(
                    "bind",
                    user_id=1,
                    chat_id=42,
                    thread_id=100,
                    window_id="@7",
                ),
                client=MagicMock(),
            )
        assert not result["ok"]
        assert "not confirmed live" in result["detail"]
        assert thread_router.get_window_for_chat_thread(42, 100) is None

    async def test_bind_unknown_presence_refused(self) -> None:
        with (
            patch(
                "ccgram.multiplexer.reconciliation.window_presence",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("ccgram.multiplexer.multiplexer"),
        ):
            result = await execute_admin_command(
                _record(
                    "bind",
                    user_id=1,
                    chat_id=42,
                    thread_id=100,
                    window_id="herdr-session-v1-truncated",
                ),
                client=MagicMock(),
            )
        assert not result["ok"]
        assert "not confirmed live" in result["detail"]
        assert thread_router.get_window_for_chat_thread(42, 100) is None

    async def test_bind_missing_argument(self) -> None:
        result = await execute_admin_command(
            _record("bind", user_id=1), client=MagicMock()
        )
        assert not result["ok"]
        assert "chat_id" in result["detail"]


class TestUnbind:
    async def test_unbind_keeps_topic_by_default(self) -> None:
        thread_router.bind_thread(1, 100, "@7", chat_id=42)
        result = await execute_admin_command(
            _record("unbind", user_id=1, chat_id=42, thread_id=100),
            client=MagicMock(),
        )
        assert result["ok"], result["detail"]
        assert "topic kept" in result["detail"]
        assert thread_router.get_window_for_chat_thread(42, 100) is None

    async def test_unbind_unbound_topic_fails(self) -> None:
        result = await execute_admin_command(
            _record("unbind", user_id=1, chat_id=42, thread_id=555),
            client=MagicMock(),
        )
        assert not result["ok"]
        assert "not bound" in result["detail"]


class TestRethread:
    async def test_rethread_moves_the_binding(self) -> None:
        thread_router.bind_thread(1, 100, "@7", chat_id=42)
        result = await execute_admin_command(
            _record(
                "rethread",
                user_id=1,
                chat_id=42,
                from_thread=100,
                to_thread=200,
            ),
            client=MagicMock(),
        )
        assert result["ok"], result["detail"]
        assert thread_router.get_window_for_chat_thread(42, 200) == "@7"
        assert thread_router.get_window_for_chat_thread(42, 100) is None

    async def test_rethread_rejects_occupied_destination(self) -> None:
        thread_router.bind_thread(1, 100, "@7", chat_id=42)
        thread_router.bind_thread(1, 200, "@8", chat_id=42)
        result = await execute_admin_command(
            _record(
                "rethread",
                user_id=1,
                chat_id=42,
                from_thread=100,
                to_thread=200,
            ),
            client=MagicMock(),
        )
        assert not result["ok"]
        assert "already bound" in result["detail"]


class TestOwnershipAndTruncation:
    async def test_unbind_rejects_foreign_owner(self) -> None:
        thread_router.bind_thread(1, 100, "@7", chat_id=42)
        result = await execute_admin_command(
            _record("unbind", user_id=2, chat_id=42, thread_id=100),
            client=MagicMock(),
        )
        assert not result["ok"]
        assert "another user" in result["detail"]
        assert thread_router.get_window_for_chat_thread(42, 100) == "@7"

    async def test_rethread_rejects_foreign_owner(self) -> None:
        thread_router.bind_thread(1, 100, "@7", chat_id=42)
        result = await execute_admin_command(
            _record(
                "rethread",
                user_id=2,
                chat_id=42,
                from_thread=100,
                to_thread=200,
            ),
            client=MagicMock(),
        )
        assert not result["ok"]
        assert "another user" in result["detail"]

    def test_truncation_rereads_retained_but_dedup_skips_executed(
        self, tmp_path
    ) -> None:
        import json as _json

        path = tmp_path / "admin_commands.jsonl"
        records = [
            _json.dumps(_record("unbind", user_id=1, chat_id=2, thread_id=i)) + "\n"
            for i in range(5)
        ]
        path.write_text("".join(records))
        first, offset = read_new_commands(path, 0)
        assert len(first) == 5

        # Rotated to a smaller retained history PLUS one new command: the
        # re-read returns the retained lines, and the consumer's id dedup
        # must skip them while still executing the new one.
        new_cmd = _json.dumps(_record("sync")) + "\n"
        path.write_text("".join(records[:2]) + new_cmd)
        again, offset2 = read_new_commands(path, offset)
        assert [r["command"] for r in again] == [
            "unbind",
            "unbind",
            "sync",
        ]
        admin_mod._EXECUTED_COMMAND_IDS.update(r["id"] for r in first)
        fresh = [r for r in again if r["id"] not in admin_mod._EXECUTED_COMMAND_IDS]
        assert [r["command"] for r in fresh] == ["sync"]


class TestWaitForResult:
    def test_wait_finds_matching_result(self, tmp_path) -> None:
        append_admin_result({"id": "cmd-9", "ok": True, "detail": "done"})
        record = wait_for_result("cmd-9", timeout=1.0)
        assert record is not None and record["ok"]

    def test_wait_times_out(self) -> None:
        assert wait_for_result("nope", timeout=0.2) is None


class TestConsume:
    async def test_consume_executes_and_appends_result(self, tmp_path) -> None:
        command_id = submit_admin_command(
            "unbind", user_id=1, chat_id=42, thread_id=100
        )
        thread_router.bind_thread(1, 100, "@7", chat_id=42)
        new_offset = await admin_mod.consume_admin_commands(MagicMock(), 0)
        assert new_offset > 0
        results = [
            json.loads(line)
            for line in (tmp_path / "admin_results.jsonl").read_text().splitlines()
        ]
        assert [r["id"] for r in results] == [command_id]
        assert results[0]["ok"]
        # Idempotent: nothing new on the second pass.
        assert await admin_mod.consume_admin_commands(MagicMock(), new_offset) == (
            new_offset
        )
