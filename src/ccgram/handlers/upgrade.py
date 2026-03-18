"""Upgrade handler — /upgrade command for self-updating ccgram via uv.

Runs `uv tool upgrade ccgram`, reports the result, and restarts the bot
process via os.execv() if an upgrade was installed. Existing tmux windows
are untouched since only the bot process restarts.

Key function: upgrade_command().
"""

import asyncio
import structlog
import re

from telegram import Update
from telegram.ext import ContextTypes

from ..config import config
from .message_sender import safe_edit, safe_reply

logger = structlog.get_logger()

_UPGRADE_TIMEOUT = 90

# Match "Upgraded ccgram v0.2.0 -> v0.2.1" or similar uv output
_VERSION_RE = re.compile(r"v(\d+\.\d+\S*)\s*$", re.MULTILINE)


async def upgrade_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /upgrade — upgrade ccgram via uv and restart if needed."""
    user = update.effective_user
    if not user or not update.message:
        return
    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    from .. import __version__

    msg = await update.message.reply_text("\u23f3 Checking for updates...")

    try:
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "tool",
            "upgrade",
            "ccgram",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        await safe_edit(
            msg, "\u274c `uv` not found. Is ccgram installed via `uv tool install`?"
        )
        return
    except OSError as exc:
        await safe_edit(msg, f"\u274c Upgrade failed: {exc}")
        return

    try:
        async with asyncio.timeout(_UPGRADE_TIMEOUT):
            stdout, stderr = await proc.communicate()
    except TimeoutError:
        proc.kill()
        await safe_edit(msg, "\u274c Upgrade timed out after 60s.")
        return

    output = (stdout or b"").decode() + (stderr or b"").decode()

    if proc.returncode != 0:
        detail = output.strip()[:200] if output.strip() else "unknown error"
        await safe_edit(
            msg, f"\u274c Upgrade failed (exit {proc.returncode}):\n`{detail}`"
        )
        return

    # Detect whether an upgrade actually happened
    # uv output: "Nothing to upgrade" when up-to-date,
    # "Upgraded ccgram v0.2.0 -> v0.2.1" when upgraded
    if "nothing to upgrade" in output.lower():
        await safe_edit(msg, f"\u2705 Already up to date (v{__version__}).")
        return

    # Parse new version from uv upgrade output
    match = _VERSION_RE.search(output)
    version_text = f"v{match.group(1)}" if match else "new version"

    await safe_edit(msg, f"\u2705 Upgraded to {version_text}. Restarting...")
    logger.info(
        "Upgrade complete (%s -> %s), scheduling restart", __version__, version_text
    )

    # Set restart flag and stop the application
    from .. import main as main_module

    main_module._restart_requested = True
    context.application.stop_running()
