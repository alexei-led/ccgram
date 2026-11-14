import json

import pytest

from ccgram.topic_mark_journal import (
    TOPIC_MARK_OPERATIONS_KEY,
    TopicMarkOperation,
    has_topic_mark_conflict,
    parse_topic_mark_operations,
    read_topic_mark_operations,
)


def test_optional_journal_has_no_migration_or_selected_state():
    assert (
        parse_topic_mark_operations({"window_states": {"@1": {"session_id": "old"}}})
        == ()
    )


def test_journal_round_trip_preserves_opaque_owners_and_route_collision():
    operations = (
        TopicMarkOperation("@1", "owner-O", False),
        TopicMarkOperation("@1", "owner-P", True),
        TopicMarkOperation("@1", None, False),
    )
    assert (
        parse_topic_mark_operations(
            {
                TOPIC_MARK_OPERATIONS_KEY: [
                    operation.to_dict() for operation in operations
                ]
            }
        )
        == operations
    )


@pytest.mark.parametrize(
    "rows",
    [
        None,
        {},
        [None],
        [{}],
        [{"window_id": "@1", "owner": "owner", "desired": 1}],
        [{"window_id": "", "owner": "owner", "desired": False}],
        [{"window_id": "@1", "owner": "", "desired": False}],
        [{"window_id": "@1", "owner": 1, "desired": False}],
        [{"window_id": "@1", "owner": "owner", "desired": False, "selected": True}],
    ],
)
def test_malformed_intent_is_not_silently_discarded(rows):
    with pytest.raises(ValueError):
        parse_topic_mark_operations({TOPIC_MARK_OPERATIONS_KEY: rows})


@pytest.mark.parametrize(
    "operations",
    [
        [
            TopicMarkOperation("@1", "same", True),
            TopicMarkOperation("@2", "same", False),
        ],
        [TopicMarkOperation("ABC", None, True), TopicMarkOperation("abc", None, False)],
    ],
)
def test_ambiguous_duplicate_keys_are_rejected(operations):
    with pytest.raises(ValueError, match="Duplicate"):
        parse_topic_mark_operations(
            {
                TOPIC_MARK_OPERATIONS_KEY: [
                    operation.to_dict() for operation in operations
                ]
            }
        )


def test_owner_keys_and_comparisons_are_case_sensitive():
    upper = TopicMarkOperation("@1", "Owner", False)
    lower = TopicMarkOperation("@1", "owner", True)
    assert upper.key != lower.key
    assert not has_topic_mark_conflict([upper], "@1", "owner", True)
    assert has_topic_mark_conflict([upper], "@other", "Owner", True)


@pytest.mark.parametrize(
    "window_id,owner,desired,conflict",
    [
        ("@1", "new-owner", True, False),
        ("@other", "old-owner", True, True),
        ("@other", "old-owner", False, False),
        ("@1", None, True, True),
        ("@other", None, True, False),
    ],
)
def test_manual_conflict_uses_owner_before_routing_key(
    window_id, owner, desired, conflict
):
    operation = TopicMarkOperation("@1", "old-owner", False)
    assert has_topic_mark_conflict([operation], window_id, owner, desired) is conflict


def test_reader_does_not_mutate_user_state(tmp_path):
    path = tmp_path / "state.json"
    content = json.dumps(
        {
            "window_states": {"@1": {"window_name": "user label"}},
            TOPIC_MARK_OPERATIONS_KEY: [
                {"window_id": "@1", "owner": "owner", "desired": False}
            ],
        }
    )
    path.write_text(content, encoding="utf-8")
    assert read_topic_mark_operations(path) == (
        TopicMarkOperation("@1", "owner", False),
    )
    assert path.read_text(encoding="utf-8") == content
    assert read_topic_mark_operations(tmp_path / "absent.json") == ()


def test_reader_fails_closed_on_corrupt_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_topic_mark_operations(path)
