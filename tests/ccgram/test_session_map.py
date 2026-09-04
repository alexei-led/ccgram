def test_pruning_keeps_a_case_variant_live_entry() -> None:
    """A hook-written key and a backend listing can spell one UUID differently.

    Deleting the entry loses the session's provider, cwd and transcript path
    for a session that is still running.
    """
    from unittest.mock import patch

    from ccgram.session_map import _dead_session_map_entries

    # agterm: its ids are the UUIDs whose case can differ, and tmux's @N form
    # has no case for this to matter to.
    with patch("ccgram.session_map.config") as cfg:
        cfg.multiplexer_name = "agterm"
        with patch("ccgram.session_map.session_map_prefix", return_value="agterm:"):
            raw = {"agterm:9f1c2d3e-4a5b": {"session_id": "s1"}}
            assert _dead_session_map_entries(raw, {"9F1C2D3E-4A5B"}) == []
            assert _dead_session_map_entries(raw, set()) != []


def test_existing_state_spelling_precedes_case_variant_binding() -> None:
    from ccgram.session import session_manager
    from ccgram.session_map import _resolve_existing_window_id
    from ccgram.thread_router import thread_router
    from ccgram.window_state_store import WindowState

    session_manager.window_states["abc-def"] = WindowState(
        session_id="old", cwd="/repo", approval_mode="yolo"
    )
    thread_router.bind_thread(100, 7, "ABC-DEF")
    try:
        assert _resolve_existing_window_id("ABC-DEF") == "abc-def"
    finally:
        session_manager.window_states.pop("abc-def", None)
        thread_router.unbind_thread(100, 7)
