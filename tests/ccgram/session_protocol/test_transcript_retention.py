"""Large tool output stays in the transcript; the index keeps only what it needs."""

import gc
import json
import tracemalloc
from pathlib import Path

import pytest

from ccgram.session_protocol.messages import SourceChangedError
from ccgram.session_protocol.transcript_source import TranscriptSource

OUTPUT_LINE = "line of tool output\n"


def _cycle(i: int, output: str) -> list[str]:
    user = {
        "type": "user",
        "uuid": f"u{i}",
        "message": {"role": "user", "content": f"question {i} " + "x" * 300},
    }
    call = {
        "type": "assistant",
        "uuid": f"a{i}",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"answer {i}"},
                {
                    "type": "tool_use",
                    "id": f"t{i}",
                    "name": "Bash",
                    "input": {"command": "ls -la"},
                },
            ],
        },
    }
    result = {
        "type": "user",
        "uuid": f"r{i}",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": output}
            ],
        },
    }
    return [json.dumps(row) + "\n" for row in (user, call, result)]


def _write(path: Path, cycles: int, output: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for i in range(cycles):
            stream.writelines(_cycle(i, output))


def _source(store, path: Path) -> TranscriptSource:
    return TranscriptSource(
        "conv", path, "conv", "claude", store.transcript_cursor("conv")
    )


async def test_history_reads_a_spilled_tool_output_from_the_transcript(store, tmp_path):
    path = tmp_path / "claude.jsonl"
    output = OUTPUT_LINE * 300
    _write(path, cycles=3, output=output)
    source = _source(store, path)
    await source.sync(store)
    assert source.builder is not None
    with source.reading():
        message = source.builder.message("conv/a1")
    assert message.parts[1].output == {"text": output, "truncated": False}


async def test_rewritten_tail_after_sync_raises_until_the_source_resyncs(
    store, tmp_path
):
    path = tmp_path / "claude.jsonl"
    _write(path, cycles=3, output=OUTPUT_LINE * 300)
    source = _source(store, path)
    await source.sync(store)
    assert source.builder is not None
    data = path.read_bytes()
    last = data.rfind(b"line of tool output")
    path.write_bytes(data[:last] + b"LINE OF TOOL OUTPUT" + data[last + 19 :])

    with pytest.raises(SourceChangedError), source.reading():
        source.builder.message("conv/a2")

    await source.sync(store)
    assert source.generation == 1
    assert source.builder is not None
    with source.reading():
        rewritten = source.builder.message("conv/g1/a2").parts[1].output["text"]
    assert rewritten.endswith("LINE OF TOOL OUTPUT\n")


async def test_index_retains_far_less_than_a_transcript_of_large_outputs(
    store, tmp_path
):
    path = tmp_path / "claude.jsonl"
    _write(path, cycles=1200, output=OUTPUT_LINE * 300)
    size = path.stat().st_size
    source = _source(store, path)
    tracemalloc.start()
    try:
        await source.sync(store)
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert retained < size * 0.5, f"retained {retained} of {size} bytes"


async def test_gemini_document_larger_than_a_marker_still_pins(store, tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"messages": [], "pad": "x" * 8000}), encoding="utf-8")
    source = TranscriptSource(
        "conv", path, "conv", "gemini", store.transcript_cursor("conv")
    )
    await source.sync(store)
    with source.reading():
        pass
