# PR title
test: avoid importing optional edge-tts at test-collection time

# PR body
## Linked Issue

Closes #<issue-number>

## What This Changes

Makes the optional `edge-tts` TTS backend truly optional for the test suite.
`tests/ccgram/tts/test_edge.py::test_synthesize_wraps_edge_exceptions_as_tts_error`
used to import `WebSocketError` from the optional `edge_tts` package at
module load time. That made `make test` fail in environments where the
`tts` extra is not installed, even though the package is explicitly
documented as optional.

The test now uses a local fake exception class and swaps
`ccgram.tts.edge._EDGE_TTS_ERRORS` via `monkeypatch` so the same wrapping
contract is verified without importing the optional dependency at
collection time.

## Checklist

- [x] There is a linked issue above
- [x] `make test` passes
- [x] `make lint` passes
- [x] This PR changes one thing only
- [x] I added a test for the new behavior or the bug fix

## Notes
- Scope is intentionally limited to the test file. No production code in
  `src/ccgram/tts/edge.py` is changed.
- The `tts` extra in `pyproject.toml` (`edge-tts>=7.2.8`) is unchanged.