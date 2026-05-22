"""Application entry point — Click CLI dispatcher and bot bootstrap.

The ``main()`` function invokes the Click command group defined in cli.py,
which dispatches to subcommands (run, hook, status, doctor).
``run_bot()`` contains the actual bot startup logic, called by the ``run``
command after CLI flags have been applied to the environment.

Module-level imports stay minimal on purpose: ``run_bot``,
``start_miniapp_if_enabled``, and ``stop_miniapp_if_enabled`` lazy-load
``config``, ``utils``, ``tmux_manager``, ``bot.create_bot``, and
``miniapp`` so that ``ccgram doctor`` / ``status`` / ``hook`` (which
import ``main`` only for ``_shutdown_signal`` and ``__version__``) do
not pay PTB or aiohttp startup cost.
"""

import asyncio
import logging
import os
import signal
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from aiohttp import web

# Set by the upgrade handler to trigger os.execv() after run_polling() returns
_restart_requested = False

# Tracks which signal triggered shutdown (0 = none/clean exit)
_shutdown_signal = 0

# Mini App server runner — populated when CCGRAM_MINIAPP_BASE_URL is set.
_miniapp_runner: "web.AppRunner | None" = None


def _on_signal(signum: int) -> None:
    """Record the shutdown signal and raise SystemExit to stop PTB.

    Registered via ``loop.add_signal_handler`` so it runs as an event-loop
    callback at a safe point — never injected at an arbitrary bytecode
    boundary inside a background task's sync I/O (which made the SystemExit
    surface as a "Background task failed" traceback and left PTB running).
    Mirrors PTB's own ``_raise_system_exit`` mechanism; the SystemExit
    propagates out of ``loop.run_forever()`` into PTB's graceful shutdown.
    """
    global _shutdown_signal
    _shutdown_signal = signum
    sig_name = signal.Signals(signum).name
    sys.stderr.write(f"\n[ccgram] {sig_name} received (pid={os.getpid()})\n")
    sys.stderr.flush()
    raise SystemExit


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Install asyncio signal handlers that trigger a graceful PTB shutdown.

    PTB's default signal handling catches SIGINT/SIGTERM/SIGABRT and exits
    with code 0 after graceful shutdown.  The restart.sh supervisor needs the
    real signal exit code (130 for SIGINT=restart, 131 for SIGQUIT=stop), so
    we register our own loop signal handlers (recording the signum) and tell
    PTB not to override them via ``stop_signals=None``.
    """
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        loop.add_signal_handler(sig, _on_signal, sig)


def _reraise_shutdown_signal() -> None:
    """Re-raise the original signal with default disposition.

    This makes the process exit with the correct code (128 + signum) so the
    parent (restart.sh) can distinguish restart from stop.
    """
    if _shutdown_signal:
        signal.signal(_shutdown_signal, signal.SIG_DFL)
        os.kill(os.getpid(), _shutdown_signal)


_GRAY_ON = "\x1b[90m"
_GRAY_OFF = "\x1b[39m"
_DEBUG_RESERVED_KEYS = frozenset(
    {"event", "level", "timestamp", "logger", "logger_name"}
)


def _dim_debug_event(
    _logger: object, _method: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    """Fold a debug line's event + kv pairs into one gray-ANSI run.

    ConsoleRenderer colors kv keys/values via its column styles (cyan/magenta),
    which it applies globally — there's no per-level kv coloring API. To make
    *the whole* debug line recede uniformly, we pre-render the kv pairs into
    the event string and drop them from the dict so ConsoleRenderer renders
    only the level chip + our pre-styled event.
    """
    if event_dict.get("level") != "debug":
        return event_dict
    event = event_dict.pop("event", "")
    parts = [str(event)] if event else []
    for key in list(event_dict):
        if key in _DEBUG_RESERVED_KEYS:
            continue
        parts.append(f"{key}={event_dict.pop(key)}")
    event_dict["event"] = f"{_GRAY_ON}{' '.join(parts)}{_GRAY_OFF}"
    return event_dict


def _use_colors(stream: object) -> bool:
    """Colorize only on a TTY, honoring the NO_COLOR / FORCE_COLOR conventions.

    Keeps raw ANSI escapes out of redirected/piped log files while still
    coloring the interactive tmux pane the daemon runs in.
    """
    # Presence-based, per the NO_COLOR / FORCE_COLOR conventions: set with any
    # value (including empty) counts. NO_COLOR wins over FORCE_COLOR.
    if "NO_COLOR" in os.environ:
        return False
    if "FORCE_COLOR" in os.environ:
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _log_level_styles() -> dict[str, str]:
    """Per-level colors so anomalies stand out and routine debug recedes.

    Values are raw ANSI escapes, matching structlog's own defaults. The key
    change from the defaults: debug is grey (was green, indistinct from info),
    and warning/error/critical are bold so they punctuate the stream.
    """
    styles = structlog.dev.ConsoleRenderer.get_default_level_styles().copy()
    styles.update(
        {
            "debug": _GRAY_ON,  # grey — recedes on a dark terminal
            "info": "\x1b[32m",  # green — normal flow (structlog default)
            "warning": "\x1b[1;33m",  # bold yellow
            "warn": "\x1b[1;33m",
            "error": "\x1b[1;31m",  # bold red
            "exception": "\x1b[1;31m",  # bold red
            "critical": "\x1b[1;91m",  # bold bright red
        }
    )
    return styles


def setup_logging(log_level: str) -> None:
    """Configure structured, colored logging for interactive CLI use."""
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    level_styles = _log_level_styles()
    stdout_colors = _use_colors(sys.stdout)
    stderr_colors = _use_colors(sys.stderr)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _dim_debug_event,
            structlog.dev.ConsoleRenderer(
                colors=stdout_colors,
                pad_event=40,
                # level_styles colors the level even when colors=False, so gate
                # it on the same flag to keep ANSI out of redirected output.
                level_styles=level_styles if stdout_colors else None,
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging for third-party libs
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(
                colors=stderr_colors,
                level_styles=level_styles if stderr_colors else None,
            ),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            ],
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    logging.getLogger("ccgram").setLevel(numeric_level)
    for name in ("httpx", "httpcore", "telegram.ext"):
        logging.getLogger(name).setLevel(logging.WARNING)


def run_bot() -> None:
    """Start the bot. Called by the ``run`` Click command after env is set."""
    log_level = (
        os.environ.get("CCGRAM_LOG_LEVEL")
        or os.environ.get("CCBOT_LOG_LEVEL")
        or "INFO"
    ).upper()
    setup_logging(log_level)

    # --- Auto-detect tmux session (before config import) ---
    explicit_session = os.environ.get("TMUX_SESSION_NAME")
    auto_detected = False

    if not explicit_session and os.environ.get("TMUX"):
        # Lazy: utils import deferred to avoid loading config-aware helpers
        # before TMUX_SESSION_NAME is set; check_duplicate_ccgram pings tmux.
        from .utils import check_duplicate_ccgram, detect_tmux_context

        detected, own_wid = detect_tmux_context()
        if detected:
            os.environ["TMUX_SESSION_NAME"] = detected
            auto_detected = True

        dup = check_duplicate_ccgram(detected or "ccgram")
        if dup:
            print(f"Error: {dup}", file=sys.stderr)
            sys.exit(1)
    else:
        own_wid = None

    try:
        # Lazy: config validates env at import time; deferring lets us catch
        # ValueError and emit a friendly error before the click subcommand layer.
        from .config import config
    except ValueError as e:
        # Lazy: ccgram_dir resolves the config dir without instantiating
        # `config` (which already failed); needed to print the .env path hint.
        from .utils import ccgram_dir

        config_dir = ccgram_dir()
        env_path = config_dir / ".env"
        print(f"Error: {e}\n")
        print(f"Create {env_path} with the following content:\n")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("  ALLOWED_USERS=your_telegram_user_id")
        print()
        print("Get your bot token from @BotFather on Telegram.")
        print("Get your user ID from @userinfobot on Telegram.")
        sys.exit(1)

    if own_wid:
        config.own_window_id = own_wid

    logger = structlog.get_logger()

    # Lazy: main runs `ccgram` startup; defer imports until the relevant subcommand executes
    from .tmux_manager import tmux_manager

    logger.info("Allowed users: %d configured", len(config.allowed_users))
    logger.info("Claude projects path: %s", config.claude_projects_path)

    # In auto-detect mode, session must already exist
    if auto_detected:
        session = tmux_manager.get_session()
        if not session:
            logger.error("Tmux session '%s' not found", config.tmux_session_name)
            sys.exit(1)
        logger.info("Using auto-detected tmux session '%s'", session.session_name)
    else:
        session = tmux_manager.get_or_create_session()
        logger.info("Tmux session '%s' ready", session.session_name)

    # Lazy: main runs `ccgram` startup; defer imports until the relevant subcommand executes
    from . import __version__

    dev = "+dev" if "+unknown" in __version__ or ".dev" in __version__ else ""
    logger.info("Starting ccgram %s%s", __version__, dev)
    # Lazy: main runs `ccgram` startup; defer imports until the relevant subcommand executes
    from .bot import create_bot

    # Create the loop here so signal handlers can be registered on it before
    # run_polling() blocks. PTB's __run reuses the loop set via set_event_loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = create_bot()
    _install_signal_handlers(loop)
    application.run_polling(
        allowed_updates=["message", "callback_query"],
        stop_signals=None,
    )

    if _restart_requested:
        logger.info("Restarting bot via os.execv(%s)", sys.argv)
        os.execv(sys.argv[0], sys.argv)

    _reraise_shutdown_signal()


async def start_miniapp_if_enabled() -> None:
    """Start the Mini App HTTP server when ``CCGRAM_MINIAPP_BASE_URL`` is set.

    Idempotent: a second call when already running is a no-op. Failures are
    logged and swallowed — the bot must keep running even if the optional
    server can't bind.
    """
    global _miniapp_runner

    if _miniapp_runner is not None:
        return

    # Lazy: main runs `ccgram` startup; defer imports until the relevant subcommand executes
    from .config import config

    if not config.miniapp_base_url:
        return

    logger = structlog.get_logger()
    try:
        # Lazy: miniapp depends on aiohttp; loading at module level would
        # break deployments that disable the dashboard via miniapp_base_url=None.
        from .miniapp import start_server

        _miniapp_runner = await start_server(
            bot_token=config.telegram_bot_token,
            host=config.miniapp_host,
            port=config.miniapp_port,
        )
        logger.info(
            "Mini App server started: base_url=%s host=%s port=%d",
            config.miniapp_base_url,
            config.miniapp_host,
            config.miniapp_port,
        )
    except OSError as exc:
        logger.error("Mini App server failed to bind: %s", exc)
        _miniapp_runner = None


async def stop_miniapp_if_enabled() -> None:
    """Stop the Mini App server if it was started; otherwise no-op."""
    global _miniapp_runner

    if _miniapp_runner is None:
        return

    logger = structlog.get_logger()
    try:
        # Lazy: miniapp depends on aiohttp; symmetric with start_miniapp_if_enabled.
        from .miniapp import stop_server

        await stop_server(_miniapp_runner)
    except OSError as exc:
        logger.warning("Mini App server stop raised: %s", exc)
    finally:
        _miniapp_runner = None


def main() -> None:
    """Main entry point — dispatches via Click CLI group."""
    # Lazy: main runs `ccgram` startup; defer imports until the relevant subcommand executes
    from .cli import cli

    cli()


if __name__ == "__main__":
    main()
