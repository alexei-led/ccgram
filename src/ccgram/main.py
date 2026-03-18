"""Application entry point — Click CLI dispatcher and bot bootstrap.

The ``main()`` function invokes the Click command group defined in cli.py,
which dispatches to subcommands (run, hook, status, doctor).
``run_bot()`` contains the actual bot startup logic, called by the ``run``
command after CLI flags have been applied to the environment.
"""

import logging
import os
import sys

import structlog

# Set by the upgrade handler to trigger os.execv() after run_polling() returns
_restart_requested = False


def setup_logging(log_level: str) -> None:
    """Configure structured, colored logging for interactive CLI use."""
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_event=40,
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
            processor=structlog.dev.ConsoleRenderer(colors=True),
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
        from .config import config
    except ValueError as e:
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

    from .tmux_manager import tmux_manager

    logger.info("Allowed users: %s", config.allowed_users)
    logger.info("Claude projects path: %s", config.claude_projects_path)

    # In auto-detect mode, session must already exist
    if auto_detected:
        session = tmux_manager.get_session()
        if not session:
            logger.error("Tmux session '%s' not found", config.tmux_session_name)
            sys.exit(1)
        logger.info("Auto-detected tmux session '%s'", session.session_name)
    else:
        session = tmux_manager.get_or_create_session()

    logger.info("Tmux session '%s' ready", session.session_name)

    logger.info("Starting Telegram bot...")
    from .bot import create_bot

    application = create_bot()
    application.run_polling(allowed_updates=["message", "callback_query"])

    if _restart_requested:
        logger.info("Restarting bot via os.execv(%s)", sys.argv)
        os.execv(sys.argv[0], sys.argv)


def main() -> None:
    """Main entry point — dispatches via Click CLI group."""
    from .cli import cli

    cli()


if __name__ == "__main__":
    main()
