import pytest
from ccgram.tts.base import TtsAudio, TtsSynthesisError
from ccgram.tts.edge import EdgeTtsSynthesizer


async def test_synthesize_collects_audio_chunks(monkeypatch):
    class DummyCommunicate:
        def __init__(self, text, voice, **kwargs):
            pass

        async def stream(self):
            yield {"type": "audio", "data": b"chunk1"}
            yield {"type": "audio", "data": b"chunk2"}
            yield {"type": "other", "data": b"ignored"}

    monkeypatch.setattr("ccgram.tts.edge._edge_tts_available", True)
    monkeypatch.setattr("ccgram.tts.edge.Communicate", DummyCommunicate)
    synth = EdgeTtsSynthesizer(voice="en-US-TestVoice")
    result = await synth.synthesize("Hello world")
    assert result == TtsAudio(data=b"chunk1chunk2")


async def test_synthesize_raises_on_empty_text():
    synth = EdgeTtsSynthesizer(voice="en-US-TestVoice")
    with pytest.raises(ValueError, match="empty"):
        await synth.synthesize("   ")


async def test_synthesize_wraps_edge_no_audio_as_tts_error(monkeypatch):
    class DummyCommunicate:
        def __init__(self, text, voice, **kwargs):
            pass

        async def stream(self):
            yield {"type": "metadata", "data": b""}

    monkeypatch.setattr("ccgram.tts.edge._edge_tts_available", True)
    monkeypatch.setattr("ccgram.tts.edge.Communicate", DummyCommunicate)
    synth = EdgeTtsSynthesizer(voice="en-US-TestVoice")
    with pytest.raises(TtsSynthesisError):
        await synth.synthesize("Hello")


async def test_synthesize_wraps_edge_exceptions_as_tts_error(monkeypatch):
    class FakeWebSocketError(Exception):
        pass

    class FailingCommunicate:
        def __init__(self, *a, **kw):
            pass

        async def stream(self):
            raise FakeWebSocketError("connection dropped")
            yield  # make it a generator

    monkeypatch.setattr("ccgram.tts.edge._edge_tts_available", True)
    monkeypatch.setattr("ccgram.tts.edge.Communicate", FailingCommunicate)
    monkeypatch.setattr("ccgram.tts.edge._EDGE_TTS_ERRORS", (FakeWebSocketError,))
    synth = EdgeTtsSynthesizer(voice="en-US-TestVoice")
    with pytest.raises(TtsSynthesisError):
        await synth.synthesize("Hello")
