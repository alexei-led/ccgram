"""Telegram bot handlers package — modular handler organization.

This package contains the Telegram bot handlers split by functionality:
  - callback_data: Callback data constants (CB_* prefixes)
  - message_queue: Per-user message queue management
  - message_sender: Safe message sending helpers with entity-based formatting
  - history: Message history pagination
  - directory_browser: Directory selection UI
  - interactive_ui: Interactive UI (AskUserQuestion, Permission Prompt, etc.)
  - polling_coordinator: Terminal status line polling loop
  - polling_strategies: Polling state management strategies
  - response_builder: Build paginated response messages
  - text_handler: Text message routing (UI guards, unbound/dead window, forwarding)
"""
