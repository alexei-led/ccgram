"""Telegram bot handlers package — modular handler organization.

Handlers are grouped into feature subpackages (Round 4, F1; Round 5
added ``commands/``):
  - commands/: /commands, /toolbar, slash-command forward + menu sync +
    failure probe + status snapshot fallback (Round 5 split of
    command_orchestration.py)
  - interactive/: AskUserQuestion / ExitPlanMode / Permission UI
  - live/: live terminal view, screenshots, pane callbacks
  - messaging_pipeline/: outbound message queue, routing, sender, tool batching
  - polling/: status polling coordinator, per-window tick (decide/observe/apply)
  - recovery/: dead-window recovery, /restore, /resume, transcript discovery, history
  - send/: /send file delivery, browser navigation, security validation
  - shell/: NL→command flow, prompt-marker setup, output capture
  - status/: status bubble lifecycle, status-bar actions, topic emoji updates
  - text/: text message routing (UI guards, unbound/dead window, forwarding)
  - toolbar/: /toolbar inline keyboard builder and callbacks
  - topics/: topic lifecycle, directory browser, window picker, /new command
  - voice/: voice transcription, confirm keyboard, callbacks

Top-level modules (leaves and cross-cutting concerns):
  - callback_data: CB_* callback data constants
  - callback_registry: prefix-based callback dispatch with self-registration
  - cleanup: topic teardown via TopicStateRegistry
  - hook_events: hook event dispatcher (Stop, Notification, Subagent*, Team*)
  - registry: central PTB handler registration (register_all) — the PTB wiring spine
  - response_builder: paginated response formatting
  - user_state: context.user_data string key constants
"""
