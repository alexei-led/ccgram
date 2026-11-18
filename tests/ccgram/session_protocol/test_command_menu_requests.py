"""A native selector is never a semantic answer to an agent tool question."""

import time

import pytest

from ccgram.providers.base import TerminalControl, TerminalMenu
from ccgram.session_protocol.backend import Prompt
from ccgram.session_protocol.events import EventHub
from ccgram.session_protocol.requests import PendingHook, RequestStore


@pytest.mark.parametrize("previously_bound", [False, True])
def test_command_menu_cannot_inherit_question_response_authority(previously_bound):
    requests = RequestStore(EventHub())
    body = "Select model\n1. Default\n2. Custom"
    hook = PendingHook(
        "conv",
        "question",
        "AskUserQuestion",
        {
            "questions": [
                {
                    "question": "Select model",
                    "options": [{"label": "Default"}, {"label": "Custom"}],
                }
            ],
        },
        time.time(),
    )
    requests.observe_hook("s", hook)
    if previously_bound:
        old = requests.reconcile("s", "conv", 1, Prompt(body, "AskUserQuestion"))
        assert old is not None and old.request.source == "hook"
    menu = TerminalMenu((TerminalControl("Set default", ("Enter",)),))
    current = requests.reconcile("s", "conv", 1, Prompt(body, "ClaudeModel", menu))
    assert current is not None and current.request.source == "screen"
    assert not current.option_keys
    assert not current.answer_keys
    assert not current.request.options
