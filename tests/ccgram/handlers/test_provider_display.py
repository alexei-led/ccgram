from ccgram.handlers.provider_display import provider_topic_name


def test_every_provider_is_labelled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CCGRAM_UNLABELLED_PROVIDERS", raising=False)
    assert provider_topic_name("chore", "claude") == "Claude · chore"
    assert provider_topic_name("review", "codex") == "Codex · review"


def test_unlabelled_providers_drop_the_label(monkeypatch) -> None:
    monkeypatch.setenv("CCGRAM_UNLABELLED_PROVIDERS", "claude")
    assert provider_topic_name("chore", "claude") == "chore"
    # An existing label is stripped rather than kept.
    assert provider_topic_name("Claude · chore", "claude") == "chore"
    assert provider_topic_name("review", "codex") == "Codex · review"
