import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram.mailbox import (
    Mailbox,
    Message,
    _DEFAULT_TTL,
    _BODY_SIZE_LIMIT,
    sanitize_dir_name,
    validate_no_traversal,
    _unsanitize_dir_name,
)


@pytest.fixture()
def mailbox(tmp_path: Path) -> Mailbox:
    return Mailbox(tmp_path / "mailbox")


class TestSanitizeDirName:
    def test_replaces_colon_with_equals(self):
        assert sanitize_dir_name("ccgram:@0") == "ccgram=@0"

    def test_roundtrip(self):
        original = "ccgram:@12"
        assert _unsanitize_dir_name(sanitize_dir_name(original)) == original

    def test_sanitize_dir_name_handles_colons(self):
        assert sanitize_dir_name("session:@5") == "session=@5"

    def test_validate_no_traversal_rejects_dotdot(self):
        with pytest.raises(ValueError, match="must not contain"):
            validate_no_traversal("../etc", "test")

    def test_validate_no_traversal_rejects_slash(self):
        with pytest.raises(ValueError, match="must not contain"):
            validate_no_traversal("foo/bar", "test")

    def test_validate_no_traversal_accepts_clean(self):
        validate_no_traversal("ccgram", "test")
        validate_no_traversal("@0", "test")


class TestMessage:
    def test_from_dict_to_dict_roundtrip(self):
        data = {
            "id": "123-abc",
            "from_id": "ccgram:@0",
            "to_id": "ccgram:@5",
            "type": "request",
            "body": "hello",
            "subject": "test",
            "created_at": "2026-03-29T10:00:00+00:00",
            "status": "pending",
            "ttl_minutes": 60,
        }
        msg = Message.from_dict(data)
        assert msg.from_id == "ccgram:@0"
        assert msg.to_id == "ccgram:@5"
        result = msg.to_dict()
        assert result["from_id"] == "ccgram:@0"
        assert result["body"] == "hello"

    def test_to_dict_sparse_serialization(self):
        msg = Message(id="1", from_id="a", to_id="b", type="request", body="hi")
        d = msg.to_dict()
        assert "reply_to" not in d
        assert "delivered_at" not in d
        assert "file_path" not in d
        assert "context" not in d

    def test_is_expired_false_for_fresh_message(self):
        msg = Message(
            id="1",
            from_id="a",
            to_id="b",
            type="request",
            body="hi",
            created_at=datetime.now(timezone.utc).isoformat(),
            ttl_minutes=60,
        )
        assert not msg.is_expired()

    def test_is_expired_true_for_old_message(self):
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=120)).isoformat()
        msg = Message(
            id="1",
            from_id="a",
            to_id="b",
            type="request",
            body="hi",
            created_at=old_time,
            ttl_minutes=60,
        )
        assert msg.is_expired()

    def test_is_expired_true_without_created_at(self):
        msg = Message(id="1", from_id="a", to_id="b", type="request", body="hi")
        assert msg.is_expired()

    def test_from_dict_legacy_from_to_keys(self):
        data = {
            "id": "1",
            "from": "ccgram:@0",
            "to": "ccgram:@5",
            "type": "request",
            "body": "x",
        }
        msg = Message.from_dict(data)
        assert msg.from_id == "ccgram:@0"
        assert msg.to_id == "ccgram:@5"


class TestMailboxSend:
    def test_send_creates_message_file(self, mailbox: Mailbox):
        msg = mailbox.send("ccgram:@0", "ccgram:@5", "hello", subject="test")
        assert msg.from_id == "ccgram:@0"
        assert msg.to_id == "ccgram:@5"
        assert msg.body == "hello"
        assert msg.status == "pending"
        assert msg.type == "request"

        inbox_dir = mailbox._inbox_dir("ccgram:@5")
        files = [
            f
            for f in inbox_dir.iterdir()
            if f.suffix == ".json" and not f.name.startswith(".")
        ]
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["body"] == "hello"

    def test_send_uses_default_ttl_per_type(self, mailbox: Mailbox):
        for msg_type, expected_ttl in _DEFAULT_TTL.items():
            msg = mailbox.send("a", "b", "hi", msg_type=msg_type)
            assert msg.ttl_minutes == expected_ttl

    def test_send_custom_ttl(self, mailbox: Mailbox):
        msg = mailbox.send("a", "b", "hi", ttl_minutes=999)
        assert msg.ttl_minutes == 999

    def test_send_invalid_type_raises(self, mailbox: Mailbox):
        with pytest.raises(ValueError, match="Invalid message type"):
            mailbox.send("a", "b", "hi", msg_type="invalid")

    def test_send_body_size_limit(self, mailbox: Mailbox):
        big_body = "x" * (_BODY_SIZE_LIMIT + 1)
        with pytest.raises(ValueError, match="exceeds"):
            mailbox.send("a", "b", big_body)

    def test_send_with_file(self, mailbox: Mailbox, tmp_path: Path):
        payload = tmp_path / "payload.txt"
        payload.write_text("large content here")
        msg = mailbox.send("a", "b", "", file_path=str(payload))
        assert msg.body.startswith("[file:")
        assert msg.file_path == str(payload)

    def test_send_with_missing_file_raises(self, mailbox: Mailbox):
        with pytest.raises(FileNotFoundError):
            mailbox.send("a", "b", "", file_path="/nonexistent/file.txt")

    def test_send_with_context(self, mailbox: Mailbox):
        ctx = {"cwd": "/project", "branch": "main", "provider": "claude"}
        msg = mailbox.send("a", "b", "hi", context=ctx)
        assert msg.context == ctx

    def test_send_reply_type(self, mailbox: Mailbox):
        msg = mailbox.send("a", "b", "hi", msg_type="reply", reply_to="orig-123")
        assert msg.type == "reply"
        assert msg.reply_to == "orig-123"


class TestMailboxQualifiedIds:
    def test_ccgram_qualified_id(self, mailbox: Mailbox):
        mailbox.send("ccgram:@0", "ccgram:@0", "self-msg")
        inbox_dir = mailbox.base_dir / "ccgram=@0"
        assert inbox_dir.is_dir()

    def test_no_bare_window_id_dirs(self, mailbox: Mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "hi")
        dirs = [d.name for d in mailbox.base_dir.iterdir() if d.is_dir()]
        assert "@5" not in dirs
        assert "ccgram=@5" in dirs


class TestMailboxInbox:
    def test_inbox_returns_fifo_order(self, mailbox: Mailbox):
        m1 = mailbox.send("a", "ccgram:@0", "first")
        time.sleep(0.01)
        m2 = mailbox.send("b", "ccgram:@0", "second")
        messages = mailbox.inbox("ccgram:@0")
        assert len(messages) == 2
        assert messages[0].id == m1.id
        assert messages[1].id == m2.id

    def test_inbox_empty_for_unknown_window(self, mailbox: Mailbox):
        assert mailbox.inbox("ccgram:@99") == []

    def test_inbox_filters_expired(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "expires fast", ttl_minutes=0)
        time.sleep(0.01)
        messages = mailbox.inbox("ccgram:@0")
        assert len(messages) == 0

    def test_inbox_filters_read_messages(self, mailbox: Mailbox):
        msg = mailbox.send("a", "ccgram:@0", "will be read")
        mailbox.read(msg.id, "ccgram:@0")
        messages = mailbox.inbox("ccgram:@0")
        assert len(messages) == 0

    def test_inbox_includes_delivered(self, mailbox: Mailbox):
        msg = mailbox.send("a", "ccgram:@0", "delivered msg")
        mailbox.mark_delivered(msg.id, "ccgram:@0")
        messages = mailbox.inbox("ccgram:@0")
        assert len(messages) == 1
        assert messages[0].status == "delivered"


class TestMailboxOrdering:
    def test_fifo_by_timestamp_prefix(self, mailbox: Mailbox):
        msgs = []
        for i in range(5):
            msgs.append(mailbox.send("a", "ccgram:@0", f"msg-{i}"))
            time.sleep(0.01)
        inbox = mailbox.inbox("ccgram:@0")
        assert [m.body for m in inbox] == [f"msg-{i}" for i in range(5)]


class TestAtomicWriteSafety:
    def test_partial_write_leaves_no_corrupt_file(self, mailbox: Mailbox):
        with (
            patch("ccgram.mailbox.json.dump", side_effect=OSError("disk full")),
            pytest.raises(OSError),
        ):
            mailbox.send("a", "ccgram:@0", "should fail")

        inbox_dir = mailbox._inbox_dir("ccgram:@0")
        if inbox_dir.exists():
            json_files = [
                f
                for f in inbox_dir.iterdir()
                if f.suffix == ".json" and not f.name.startswith(".")
            ]
            assert len(json_files) == 0

    def test_concurrent_read_during_write(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "existing")
        mailbox.send("b", "ccgram:@0", "another")
        messages = mailbox.inbox("ccgram:@0")
        assert len(messages) == 2


class TestTTLExpiration:
    def test_per_type_defaults(self):
        assert _DEFAULT_TTL["request"] == 60
        assert _DEFAULT_TTL["reply"] == 120
        assert _DEFAULT_TTL["notify"] == 240
        assert _DEFAULT_TTL["broadcast"] == 480

    def test_expired_message_not_in_inbox(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "expires", ttl_minutes=0)
        time.sleep(0.01)
        assert mailbox.inbox("ccgram:@0") == []

    def test_non_expired_message_in_inbox(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "stays", ttl_minutes=9999)
        assert len(mailbox.inbox("ccgram:@0")) == 1


class TestMessageStatusTransitions:
    def test_pending_to_read(self, mailbox: Mailbox):
        msg = mailbox.send("a", "ccgram:@0", "hello")
        assert msg.status == "pending"
        updated = mailbox.read(msg.id, "ccgram:@0")
        assert updated is not None
        assert updated.status == "read"
        assert updated.read_at is not None

    def test_pending_to_delivered(self, mailbox: Mailbox):
        msg = mailbox.send("a", "ccgram:@0", "hello")
        updated = mailbox.mark_delivered(msg.id, "ccgram:@0")
        assert updated is not None
        assert updated.status == "delivered"
        assert updated.delivered_at is not None

    def test_pending_to_replied(self, mailbox: Mailbox):
        msg = mailbox.send("ccgram:@5", "ccgram:@0", "question?")
        reply = mailbox.reply(msg.id, "ccgram:@0", "answer!")
        assert reply is not None
        assert reply.type == "reply"
        assert reply.reply_to == msg.id
        assert reply.from_id == "ccgram:@0"
        assert reply.to_id == "ccgram:@5"

        original, _ = mailbox._find_message(msg.id, "ccgram:@0")
        assert original is not None
        assert original.status == "replied"

    def test_read_nonexistent_returns_none(self, mailbox: Mailbox):
        assert mailbox.read("nonexistent", "ccgram:@0") is None

    def test_reply_nonexistent_returns_none(self, mailbox: Mailbox):
        assert mailbox.reply("nonexistent", "ccgram:@0", "hi") is None


class TestFileSupport:
    def test_file_body_references_path(self, mailbox: Mailbox, tmp_path: Path):
        payload = tmp_path / "data.txt"
        payload.write_text("big data")
        msg = mailbox.send("a", "b", "", file_path=str(payload))
        assert f"[file:{payload}]" in msg.body

    def test_body_size_enforcement(self, mailbox: Mailbox):
        exact_limit = "x" * _BODY_SIZE_LIMIT
        msg = mailbox.send("a", "b", exact_limit)
        assert msg.body == exact_limit

        over_limit = "x" * (_BODY_SIZE_LIMIT + 1)
        with pytest.raises(ValueError, match="exceeds"):
            mailbox.send("a", "b", over_limit)


class TestFileNotFoundResilience:
    def test_sweep_handles_concurrent_delete(self, mailbox: Mailbox):
        msg = mailbox.send("a", "ccgram:@0", "will expire", ttl_minutes=0)
        time.sleep(0.01)

        msg_path = mailbox._inbox_dir("ccgram:@0") / f"{msg.id}.json"
        msg_path.unlink()

        removed = mailbox.sweep("ccgram:@0")
        assert removed == 0

    def test_inbox_skips_corrupt_json(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "good")
        inbox_dir = mailbox._inbox_dir("ccgram:@0")
        corrupt = inbox_dir / "0000000000-corrupt.json"
        corrupt.write_text("{invalid json")
        messages = mailbox.inbox("ccgram:@0")
        assert len(messages) == 1


class TestPruneDead:
    def test_removes_dead_window_dirs(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "msg1")
        mailbox.send("a", "ccgram:@5", "msg2")
        mailbox.send("a", "ccgram:@9", "msg3")

        removed = mailbox.prune_dead({"ccgram:@0"})
        assert removed == 2
        assert mailbox._inbox_dir("ccgram:@0").is_dir()
        assert not mailbox._inbox_dir("ccgram:@5").exists()
        assert not mailbox._inbox_dir("ccgram:@9").exists()

    def test_noop_on_empty_mailbox(self, mailbox: Mailbox):
        assert mailbox.prune_dead(set()) == 0


class TestPendingUndelivered:
    def test_returns_old_pending_messages(self, mailbox: Mailbox):
        msg = mailbox.send("a", "ccgram:@0", "stuck")
        results = mailbox.pending_undelivered(min_age_seconds=0)
        assert len(results) == 1
        assert results[0].id == msg.id

    def test_ignores_recent_messages(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "just sent")
        results = mailbox.pending_undelivered(min_age_seconds=9999)
        assert len(results) == 0

    def test_ignores_delivered_messages(self, mailbox: Mailbox):
        msg = mailbox.send("a", "ccgram:@0", "delivered")
        mailbox.mark_delivered(msg.id, "ccgram:@0")
        results = mailbox.pending_undelivered(min_age_seconds=0)
        assert len(results) == 0

    def test_ignores_expired_messages(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "expires", ttl_minutes=0)
        time.sleep(0.01)
        results = mailbox.pending_undelivered(min_age_seconds=0)
        assert len(results) == 0

    def test_empty_base_dir(self, mailbox: Mailbox):
        assert mailbox.pending_undelivered() == []


class TestMigrateIds:
    def test_renames_inbox_dirs(self, mailbox: Mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello")
        mailbox.migrate_ids({"ccgram:@5": "ccgram:@7"})
        assert not mailbox._inbox_dir("ccgram:@5").exists()
        assert mailbox._inbox_dir("ccgram:@7").is_dir()

    def test_updates_message_ids_inside(self, mailbox: Mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello")
        mailbox.migrate_ids({"ccgram:@5": "ccgram:@7"})

        msgs = mailbox.inbox("ccgram:@7")
        assert len(msgs) == 1
        assert msgs[0].to_id == "ccgram:@7"

    def test_skips_if_target_exists(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@5", "old")
        mailbox.send("a", "ccgram:@7", "existing")
        mailbox.migrate_ids({"ccgram:@5": "ccgram:@7"})
        assert mailbox._inbox_dir("ccgram:@5").is_dir()
        assert mailbox._inbox_dir("ccgram:@7").is_dir()


class TestBroadcast:
    def test_broadcast_to_multiple_recipients(self, mailbox: Mailbox):
        sent = mailbox.broadcast("ccgram:@0", ["ccgram:@5", "ccgram:@8"], "hello all")
        assert len(sent) == 2
        assert len(mailbox.inbox("ccgram:@5")) == 1
        assert len(mailbox.inbox("ccgram:@8")) == 1
        assert len(mailbox.inbox("ccgram:@0")) == 0

    def test_broadcast_type_is_broadcast(self, mailbox: Mailbox):
        sent = mailbox.broadcast("ccgram:@0", ["ccgram:@5"], "typed")
        assert sent[0].type == "broadcast"

    def test_broadcast_default_ttl_480(self, mailbox: Mailbox):
        sent = mailbox.broadcast("ccgram:@0", ["ccgram:@5"], "ttl check")
        assert sent[0].ttl_minutes == 480

    def test_broadcast_custom_ttl(self, mailbox: Mailbox):
        sent = mailbox.broadcast("ccgram:@0", ["ccgram:@5"], "custom", ttl_minutes=120)
        assert sent[0].ttl_minutes == 120

    def test_broadcast_with_subject(self, mailbox: Mailbox):
        sent = mailbox.broadcast(
            "ccgram:@0", ["ccgram:@5"], "body", subject="Important"
        )
        assert sent[0].subject == "Important"

    def test_broadcast_empty_recipients(self, mailbox: Mailbox):
        sent = mailbox.broadcast("ccgram:@0", [], "nobody")
        assert sent == []

    def test_broadcast_with_file(self, mailbox: Mailbox, tmp_path: Path):
        payload = tmp_path / "broadcast.txt"
        payload.write_text("large broadcast content")
        sent = mailbox.broadcast(
            "ccgram:@0", ["ccgram:@5", "ccgram:@8"], "", file_path=str(payload)
        )
        assert len(sent) == 2
        assert all(m.body.startswith("[file:") for m in sent)

    def test_broadcast_with_context(self, mailbox: Mailbox):
        ctx = {"cwd": "/project", "branch": "main", "provider": "claude"}
        sent = mailbox.broadcast("ccgram:@0", ["ccgram:@5"], "ctx", context=ctx)
        assert sent[0].context == ctx

    def test_broadcast_continues_on_individual_failure(self, mailbox: Mailbox):
        with patch.object(
            mailbox,
            "send",
            side_effect=[
                ValueError("fail"),
                Message(
                    id="ok",
                    from_id="ccgram:@0",
                    to_id="ccgram:@8",
                    type="broadcast",
                    body="ok",
                ),
            ],
        ):
            sent = mailbox.broadcast("ccgram:@0", ["ccgram:@5", "ccgram:@8"], "partial")
        assert len(sent) == 1
        assert sent[0].to_id == "ccgram:@8"


class TestSweep:
    def test_removes_expired_messages(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "expires", ttl_minutes=0)
        time.sleep(0.01)
        removed = mailbox.sweep("ccgram:@0")
        assert removed == 1

    def test_removes_replied_messages(self, mailbox: Mailbox):
        msg = mailbox.send("ccgram:@5", "ccgram:@0", "question")
        mailbox.reply(msg.id, "ccgram:@0", "answer")
        removed = mailbox.sweep("ccgram:@0")
        assert removed == 1

    def test_preserves_pending_messages(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "fresh", ttl_minutes=9999)
        removed = mailbox.sweep("ccgram:@0")
        assert removed == 0
        assert len(mailbox.inbox("ccgram:@0")) == 1

    def test_sweep_all_windows(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "expires", ttl_minutes=0)
        mailbox.send("a", "ccgram:@5", "expires too", ttl_minutes=0)
        time.sleep(0.01)
        removed = mailbox.sweep()
        assert removed == 2

    def test_sweep_removes_corrupt_files(self, mailbox: Mailbox):
        mailbox.send("a", "ccgram:@0", "good")
        inbox_dir = mailbox._inbox_dir("ccgram:@0")
        corrupt = inbox_dir / "0000000000-bad.json"
        corrupt.write_text("not json at all")
        removed = mailbox.sweep("ccgram:@0")
        assert removed == 1
        assert not corrupt.exists()

    def test_sweep_empty_inbox(self, mailbox: Mailbox):
        assert mailbox.sweep("ccgram:@99") == 0
