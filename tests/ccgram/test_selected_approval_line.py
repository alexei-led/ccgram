from ccgram.providers.codex import CodexProvider
import pytest


@pytest.mark.parametrize("selected", [1, 2])
def test_codex_approval_retains_both_choices_after_prior_chat_prompts(selected):
    lines = [
        "› Earlier user request",
        "• Earlier response",
        "",
        "› Inspect the counter",
        "• Calling approval_probe.inspect_probe({})",
        "",
        "  Field 1/1",
        '  Allow the approval_probe MCP server to run tool "inspect_probe"?',
        ("  › " if selected == 1 else "    ") + "1. Allow   Run the tool and continue.",
        ("  › " if selected == 2 else "    ") + "2. Cancel  Cancel this tool call",
        "",
        "  enter to submit | esc to cancel",
    ]
    status = CodexProvider().parse_terminal_status("\n".join(lines))
    assert status is not None and status.is_interactive
    assert "1. Allow" in status.raw_text
    assert "2. Cancel" in status.raw_text
    assert f"› {selected}." in status.raw_text
