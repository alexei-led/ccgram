# Callback Dispatch

## Functional Responsibilities

- Maintain a prefix → handler mapping (the callback registry)
- Provide `@register(*prefixes)` decorator for handler self-registration at module import time
- Dispatch incoming callback queries to the matching handler by longest-prefix match
- Perform authorization check before dispatch (user allowed, group_id match)
- Record group chat IDs on every callback (for forum topic routing)
- Load all callback-bearing handler modules to trigger `@register` decorators

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **Prefix matching algorithm** — longest-match ordering for correctness with overlapping prefixes (e.g., `CB_SCREENSHOT` vs `CB_SCREENSHOT_STATUS`)
- **Module discovery list** — the explicit set of handler modules containing `@register` callbacks (in `load_handlers()`)
- **Authorization check logic** — user allowlist validation, group_id matching, `answer_callback_query` for unauthorized users
- **Dispatch error handling** — catch handler exceptions, answer query with error feedback
- **Registration deduplication** — prevent double-registration if a module is imported twice

## Subdomain Classification

**Generic** — The callback registry is stable infrastructure. The dispatch pattern rarely changes; what changes is which handlers register themselves. This is intentional — the registry absorbs the volatility of handler additions without changing itself.

## Integration Contracts

### ← Bot Shell (depended on by)

- **Direction**: Bot Shell depends on Callback Dispatch
- **Contract type**: Contract (registration API)
- **What is shared**: Dispatch function and handler loading trigger
- **Contract definition**:

  ```python
  def load_handlers() -> None:
      """Import all handler modules to trigger @register decorators."""
      ...

  async def dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
      """Route callback query to registered handler by longest-prefix match."""
      ...
  ```

### ← Handler modules (depended on by)

- **Direction**: Handler modules depend on Callback Dispatch (decorator import)
- **Contract type**: Contract (self-registration decorator)
- **What is shared**: Prefix strings and handler function references
- **Contract definition**:

  ```python
  def register(*prefixes: str) -> Callable:
      """Decorator to register a callback handler for given prefix strings."""
      ...

  # Usage in handler module:
  @register(CB_DIR_SELECT, CB_DIR_BACK, CB_DIR_HOME)
  async def handle_directory_callback(update, context): ...
  ```

### → Callback Data (depends on)

- **Direction**: Callback Dispatch depends on Callback Data (indirectly, via handlers)
- **Contract type**: Contract (string constants)
- **What is shared**: `CB_*` prefix constants that handlers pass to `@register`
- **Contract definition**: String constants in `handlers/callback_data.py` — the single source of truth for all callback data prefixes

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Changing the matching algorithm** (e.g., exact match instead of prefix match) — only `dispatch()` changes
- **Adding authorization logic** (e.g., admin-only callbacks) — only `dispatch()` changes
- **Adding dispatch middleware** (e.g., logging, metrics) — wrap `dispatch()` internals
- **Changing error handling strategy** — only exception handling in `dispatch()` changes

Note: Adding a new callback handler does NOT require changes to this module beyond adding one import line to `load_handlers()`. The handler self-registers via `@register`.
