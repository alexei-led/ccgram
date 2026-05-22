from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.messaging_pipeline.message_routing import handle_new_message
from ccgram.session_monitor import NewMessage


def _make_msg(
    *,
    text: str = "hello",
    content_type: str = "text",
    phase: str | None = None,
    tool_name: str = "",
    tool_use_id: str = "",
    role: str = "assistant",
    is_complete: bool = True,
    session_id: str = "sess-1",
) -> NewMessage:
    return NewMessage(
        session_id=session_id,
        text=text,
        content_type=content_type,
        phase=phase,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        role=role,
        is_complete=is_complete,
    )


@pytest.fixture
def bot() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_deps():
    with (
        patch("ccgram.handlers.messaging_pipeline.message_routing.session_query") as sq,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.enqueue_content_message",
            new_callable=AsyncMock,
        ) as eq,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.get_message_queue"
        ) as gmq,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.handle_interactive_ui",
            new=AsyncMock(return_value=False),
        ) as hui,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.set_interactive_mode"
        ) as sim,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.clear_interactive_mode"
        ) as cim,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.clear_interactive_msg",
            new_callable=AsyncMock,
        ) as cmsg,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.get_interactive_msg_id",
            return_value=None,
        ) as gimid,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.build_response_parts",
            return_value=["parts"],
        ) as brp,
        patch(
            "ccgram.handlers.messaging_pipeline.message_routing.user_preferences"
        ) as up,
    ):
        sq.find_users_for_session.return_value = [(100, "@5", 42)]
        sq.resolve_session_for_window = AsyncMock(return_value=None)
        gmq.return_value = None
        yield {
            "sq": sq,
            "eq": eq,
            "gmq": gmq,
            "hui": hui,
            "sim": sim,
            "cim": cim,
            "cmsg": cmsg,
            "gimid": gimid,
            "brp": brp,
            "up": up,
        }


async def test_no_active_users_returns_early(bot, mock_deps):
    mock_deps["sq"].find_users_for_session.return_value = []
    await handle_new_message(_make_msg(), bot)
    mock_deps["eq"].assert_not_called()


async def test_short_thinking_is_dropped(bot, mock_deps):
    await handle_new_message(_make_msg(text="hm", content_type="thinking"), bot)
    mock_deps["eq"].assert_not_called()


async def test_long_thinking_is_kept(bot, mock_deps):
    await handle_new_message(_make_msg(text="x" * 50, content_type="thinking"), bot)
    mock_deps["eq"].assert_called_once()


async def test_interactive_tool_use_handled_skips_enqueue(bot, mock_deps):
    mock_deps["hui"].return_value = True
    await handle_new_message(
        _make_msg(
            text="?",
            content_type="tool_use",
            tool_name="AskUserQuestion",
            tool_use_id="t1",
        ),
        bot,
    )
    mock_deps["sim"].assert_called_once_with(100, "@5", 42)
    mock_deps["hui"].assert_called_once()
    mock_deps["eq"].assert_not_called()


async def test_interactive_tool_use_unhandled_falls_through(bot, mock_deps):
    mock_deps["hui"].return_value = False
    await handle_new_message(
        _make_msg(
            text="?",
            content_type="tool_use",
            tool_name="AskUserQuestion",
            tool_use_id="t1",
        ),
        bot,
    )
    mock_deps["cim"].assert_called_once()
    mock_deps["eq"].assert_called_once()


async def test_pending_interactive_msg_is_cleared(bot, mock_deps):
    mock_deps["gimid"].return_value = 999
    await handle_new_message(_make_msg(text="reply"), bot)
    mock_deps["cmsg"].assert_called_once()
    mock_deps["eq"].assert_called_once()


async def test_complete_message_enqueues_content(bot, mock_deps):
    await handle_new_message(_make_msg(text="done", is_complete=True), bot)
    mock_deps["eq"].assert_called_once()
    kwargs = mock_deps["eq"].call_args.kwargs
    assert kwargs["user_id"] == 100
    assert kwargs["window_id"] == "@5"
    assert kwargs["thread_id"] == 42
