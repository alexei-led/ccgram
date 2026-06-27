"""Tests for the /split command handler."""

from unittest.mock import AsyncMock, MagicMock, patch


def _make_update(user_id=1, thread_id=42, chat_id=100):
    user = MagicMock()
    user.id = user_id
    message = MagicMock()
    message.message_thread_id = thread_id
    chat = MagicMock()
    chat.id = chat_id
    update = MagicMock()
    update.effective_user = user
    update.message = message
    update.effective_chat = chat
    return update


def _ctx(args=None):
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


async def _run(
    update, ctx, *, window_id, window_alive=True, new_pane: str | None = "w2:p2"
):
    from ccgram.handlers.split_command import split_command

    mock_reply = AsyncMock()
    mock_tr = MagicMock()
    mock_tr.get_window_for_thread.return_value = window_id
    mock_tm = MagicMock()
    mock_tm.find_window_by_id = AsyncMock(
        return_value=MagicMock() if window_alive else None
    )
    mock_tm.split_window = AsyncMock(return_value=new_pane)
    mock_tm.send_to_pane = AsyncMock(return_value=True)

    with (
        patch("ccgram.config.config") as mock_cfg,
        patch("ccgram.handlers.callback_helpers.get_thread_id", return_value=42),
        patch("ccgram.handlers.split_command.thread_router", mock_tr),
        patch("ccgram.handlers.split_command.tmux_manager", mock_tm),
        patch(
            "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
            mock_reply,
        ),
    ):
        mock_cfg.is_user_allowed.return_value = True
        await split_command(update, ctx)
    return mock_reply, mock_tm


async def test_unbound_window_sends_error() -> None:
    reply, tm = await _run(_make_update(), _ctx(), window_id=None)
    reply.assert_called_once()
    assert "not bound" in reply.call_args[0][1].lower()
    tm.split_window.assert_not_called()


async def test_dead_window_sends_error() -> None:
    reply, tm = await _run(_make_update(), _ctx(), window_id="@0", window_alive=False)
    reply.assert_called_once()
    assert "no longer exists" in reply.call_args[0][1].lower()
    tm.split_window.assert_not_called()


async def test_split_failure_sends_error() -> None:
    reply, tm = await _run(_make_update(), _ctx(), window_id="@0", new_pane=None)
    tm.split_window.assert_awaited_once_with("@0")
    reply.assert_called_once()
    assert "could not split" in reply.call_args[0][1].lower()


async def test_split_no_args_reports_new_pane() -> None:
    reply, tm = await _run(_make_update(), _ctx(), window_id="@0", new_pane="%5")
    tm.split_window.assert_awaited_once_with("@0")
    tm.send_to_pane.assert_not_called()
    text = reply.call_args[0][1]
    assert "%5" in text and "/panes" in text


async def test_split_with_command_runs_in_new_pane() -> None:
    reply, tm = await _run(
        _make_update(), _ctx(["claude", "--resume"]), window_id="@0", new_pane="%5"
    )
    tm.split_window.assert_awaited_once_with("@0")
    tm.send_to_pane.assert_awaited_once_with("%5", "claude --resume", window_id="@0")
    text = reply.call_args[0][1]
    assert "claude --resume" in text
