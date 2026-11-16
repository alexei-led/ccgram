"""Pure test witnesses are not evidence of native process provenance."""

from dataclasses import dataclass, field

from ccgram.topic_mark_types import TopicMarkWriteWitness


def make_witness(
    dispatch_id: str = "dispatch-1",
    *,
    backend: str = "tmux",
    pin_path: str | None = None,
) -> TopicMarkWriteWitness:
    return TopicMarkWriteWitness(
        dispatch_id=dispatch_id,
        backend=backend,
        host_id="host-1",
        boot_id="boot-1",
        namespace_id="darwin-local",
        native_pid=123,
        native_start="darwin:100:200",
        code_id="cdhash:abcdef",
        socket_dev=0,
        socket_ino=456,
        pin_path=pin_path,
    )


@dataclass
class RecordingTopicMarkGuard:
    armed: list[TopicMarkWriteWitness] = field(default_factory=list)
    terminal_ids: list[str] = field(default_factory=list)

    def arm(self, witness: TopicMarkWriteWitness) -> None:
        self.armed.append(witness)

    def terminal(self, dispatch_id: str) -> None:
        self.terminal_ids.append(dispatch_id)
