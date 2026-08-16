# Issue title
test: avoid importing optional edge-tts at test-collection time

# Issue body
## Problem
`tests/ccgram/tts/test_edge.py::test_synthesize_wraps_edge_exceptions_as_tts_error`
performs `from edge_tts.exceptions import WebSocketError` at module load time.
This fails when running the test suite without the optional `edge-tts`
extras installed (`pip install ccgram[tts]`), producing
`ModuleNotFoundError: No module named 'edge_tts'` and breaking `make test`
in minimal environments even though `edge_tts` is explicitly optional.

## Impact
- `make test` is not green in environments where the optional TTS extras are not installed.
- The failure is unrelated to CCGram core behavior and obscures real regressions.
- This blocks any CI configuration that runs the suite without the `tts` extra.

## Goal
Make the optional `edge-tts` dependency truly optional for the test suite so
`make test` can be green without `pip install ccgram[tts]`, while still
verifying that backend exceptions are wrapped as `TtsSynthesisError`.

## Proposed change
- In `test_synthesize_wraps_edge_exceptions_as_tts_error`, replace the
  unconditional `from edge_tts.exceptions import WebSocketError` with a local
  fake exception class.
- Use `monkeypatch.setattr("ccgram.tts.edge._EDGE_TTS_ERRORS", ...)` so the
  test verifies the wrapping contract without depending on the real
  optional package at collection time.

## Environment
- CCGram version: current `main`
- Python: 3.14
- Optional dep: `edge-tts` is declared as `tts` extra in `pyproject.toml`
  but should not be required to run the test suite