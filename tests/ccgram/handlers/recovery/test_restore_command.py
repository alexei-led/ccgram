from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.recovery.restore_command import restore_command
from ccgram.handlers.user_state import PENDING_THREAD_ID, RECOVERY_WINDOW_ID

_RS = "ccgram.handlers.recovery.restore_command"


@pytest.fixture()
def env():
    """A bound topic whose window is dead — the /restore happy path."""
    with (
        patch(f"{_RS}.thread_router") as router,
        patch(f"{_RS}.tmux_manager") as tmux,
        patch(f"{_RS}.config") as config,
        patch(f"{_RS}.window_query") as wq,
        patch(f"{_RS}.render_banner") as render,
        patch(f"{_RS}.safe_reply", new_callable=AsyncMock) as reply,
    ):
        config.is_user_allowed.return_value = True
        router.resolve_window_for_thread.return_value = "@5"
        router.get_display_name.return_value = "my-project"
        tmux.find_window_by_id = AsyncMock(return_value=None)
        wq.get_window_provider.return_value = "claude"
        render.return_value = ("⚠ Banner text", MagicMock())
        yield SimpleNamespace(
            router=router,
            tmux=tmux,
            config=config,
            wq=wq,
            render=render,
            reply=reply,
        )


def _make_update(*, user_id: int = 100, thread_id: int | None = 42):
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.message = AsyncMock()
    update.message.message_thread_id = thread_id
    update.message.chat.type = "supergroup"
    update.message.chat.id = -100999
    return update


def _make_context():
    context = MagicMock()
    context.user_data = {}
    context.bot = AsyncMock()
    return context


class TestRestoreCommandRendersTheBanner:
    async def test_renders_a_restore_banner_for_the_bound_window(
        self, env, tmp_path
    ) -> None:
        env.wq.view_window.return_value = MagicMock(cwd=str(tmp_path))

        await restore_command(_make_update(), _make_context())

        banner = env.render.call_args.args[0]
        assert banner.window_id == "@5"
        assert banner.mode == "restore"
        assert banner.cwd == str(tmp_path)
        assert banner.display == "my-project"
        assert banner.thread_id == 42
        env.reply.assert_called_once()
        assert env.reply.call_args.args[1] == "⚠ Banner text"
        assert env.reply.call_args.kwargs["reply_markup"] is not None

    async def test_display_falls_back_to_the_window_id(self, env, tmp_path) -> None:
        env.router.get_display_name.return_value = ""
        env.wq.view_window.return_value = MagicMock(cwd=str(tmp_path))

        await restore_command(_make_update(), _make_context())

        assert env.render.call_args.args[0].display == "@5"

    async def test_records_pending_recovery_state(self, env, tmp_path) -> None:
        env.wq.view_window.return_value = MagicMock(cwd=str(tmp_path))
        ctx = _make_context()

        await restore_command(_make_update(), ctx)

        assert ctx.user_data[PENDING_THREAD_ID] == 42
        assert ctx.user_data[RECOVERY_WINDOW_ID] == "@5"

    async def test_does_not_touch_windows_or_bindings(self, env, tmp_path) -> None:
        env.wq.view_window.return_value = MagicMock(cwd=str(tmp_path))

        await restore_command(_make_update(), _make_context())

        env.tmux.create_window.assert_not_called()
        env.router.bind_thread.assert_not_called()
        env.router.unbind_thread.assert_not_called()

    async def test_missing_user_data_does_not_break_the_banner(
        self, env, tmp_path
    ) -> None:
        env.wq.view_window.return_value = MagicMock(cwd=str(tmp_path))
        ctx = _make_context()
        ctx.user_data = None

        await restore_command(_make_update(), ctx)

        env.reply.assert_called_once()
        assert env.reply.call_args.args[1] == "⚠ Banner text"


class TestRestoreCommandRefusals:
    @pytest.mark.parametrize(
        "field", ["effective_user", "message"], ids=["no-user", "no-message"]
    )
    async def test_incomplete_update_is_ignored(self, env, field: str) -> None:
        update = _make_update()
        setattr(update, field, None)

        await restore_command(update, _make_context())

        env.reply.assert_not_called()

    async def test_unauthorized_user_rejected(self, env) -> None:
        env.config.is_user_allowed.return_value = False

        await restore_command(_make_update(), _make_context())

        env.reply.assert_called_once()
        assert "not authorized" in env.reply.call_args.args[1]

    async def test_outside_a_topic_rejected(self, env) -> None:
        await restore_command(_make_update(thread_id=None), _make_context())

        assert "inside a topic" in env.reply.call_args.args[1]

    async def test_unbound_topic_rejected(self, env) -> None:
        env.router.resolve_window_for_thread.return_value = None

        await restore_command(_make_update(), _make_context())

        assert "No session bound" in env.reply.call_args.args[1]

    async def test_live_window_has_nothing_to_restore(self, env) -> None:
        env.tmux.find_window_by_id = AsyncMock(return_value=MagicMock())

        await restore_command(_make_update(), _make_context())

        assert "still running" in env.reply.call_args.args[1]
        env.render.assert_not_called()

    @pytest.mark.parametrize(
        "view", [None, MagicMock(cwd="")], ids=["no-window-state", "blank-cwd"]
    )
    async def test_missing_state_is_not_reported_as_a_missing_directory(
        self, env, view
    ) -> None:
        """No window state means unknown folder, not a deleted one (#176)."""
        env.wq.view_window.return_value = view

        await restore_command(_make_update(), _make_context())

        text = env.reply.call_args.args[1]
        assert "Directory no longer exists" not in text
        assert "session state is gone" in text
        assert "/resume" in text
        env.render.assert_not_called()

    async def test_deleted_directory_keeps_the_filesystem_message(self, env) -> None:
        env.wq.view_window.return_value = MagicMock(cwd="/nonexistent/path")

        await restore_command(_make_update(), _make_context())

        assert "Directory no longer exists" in env.reply.call_args.args[1]
        env.render.assert_not_called()
