# Callback Dispatch — Test Specification

## Unit Tests

- **test_register_single_prefix**: Register handler with one prefix. Expected: handler stored in registry keyed by prefix.
- **test_register_multiple_prefixes**: Register handler with three prefixes. Expected: all three prefixes map to same handler.
- **test_register_preserves_function**: Register handler, retrieve from registry. Expected: same function object (decorator is transparent).
- **test_dispatch_matches_prefix**: Register handler for "dir:", dispatch callback with data="dir:select:/home". Expected: handler called.
- **test_dispatch_longest_prefix_match**: Register "sc:" and "sc:status:". Dispatch "sc:status:refresh". Expected: "sc:status:" handler called (longest match).
- **test_dispatch_no_match**: Dispatch callback with data="unknown:foo". Expected: query answered with "unknown" feedback, no handler called.
- **test_dispatch_noop**: Dispatch callback with data="noop". Expected: `answer_callback_query()` called, no handler invoked.
- **test_load_handlers_imports_modules**: Call `load_handlers()`. Expected: all callback-bearing handler modules imported (verify via registry population count).
- **test_load_handlers_idempotent**: Call `load_handlers()` twice. Expected: no duplicate registrations, same handler count.

## Integration Contract Tests

- **test_handler_self_registration**: Import a handler module that uses `@register(CB_DIR_SELECT)`. Expected: `CB_DIR_SELECT` prefix appears in registry mapping to that handler.
- **test_dispatch_called_as_callback_handler**: Wire `dispatch` as PTB `CallbackQueryHandler`, send callback Update. Expected: dispatch invoked, correct handler called.
- **test_callback_data_constants_resolvable**: All `CB_*` constants used in `@register` decorators are importable from `callback_data.py`. Expected: no ImportError.
- **test_authorization_check_rejects_unauthorized**: Dispatch callback from unauthorized user. Expected: `answer_callback_query` with rejection message, handler not called.
- **test_authorization_check_allows_authorized**: Dispatch callback from authorized user. Expected: handler called normally.

## Boundary Tests

- **test_empty_callback_data**: Dispatch callback with data="". Expected: no match, handled gracefully.
- **test_none_callback_data**: Dispatch callback with data=None. Expected: handled gracefully, no crash.
- **test_very_long_callback_data**: Dispatch with 64-byte callback data string. Expected: prefix matching works correctly.
- **test_handler_raises_exception**: Registered handler raises `ValueError`. Expected: exception caught, `answer_callback_query` with error feedback, exception logged.
- **test_handler_raises_telegram_error**: Handler raises `BadRequest`. Expected: Telegram error handled specifically (not generic catch).
- **test_duplicate_prefix_registration**: Two handlers register for same prefix. Expected: last registration wins (or raises error — design choice).

## Behavior Tests

- **test_directory_callback_routed**: User taps directory browser button. Expected: callback with "dir:" prefix dispatched to `handle_directory_callback`.
- **test_screenshot_callback_routed**: User taps screenshot button. Expected: callback with "sc:" prefix dispatched to `handle_screenshot_callback`.
- **test_interactive_callback_routed**: User taps permission prompt button. Expected: callback with "aq:" prefix dispatched to `handle_interactive_callback`.
- **test_new_handler_added_without_dispatch_change**: Add a new handler module with `@register("new:")`, add import to `load_handlers()`. Expected: "new:" callbacks dispatched correctly with no changes to `dispatch()`.
