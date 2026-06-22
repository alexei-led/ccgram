"""Callback data constants for Telegram inline keyboards.

Defines all CB_* prefixes used for routing callback queries in the bot.
Each prefix identifies a specific action or navigation target.

Constants:
  - CB_HISTORY_*: History pagination
  - CB_DIR_*: Directory browser navigation
  - CB_WIN_*: Window picker (bind existing unbound window)
  - CB_SCREENSHOT_*: Screenshot refresh
  - CB_ASK_*: Interactive UI navigation (arrows, enter, esc)
  - CB_SESSIONS_*: Sessions dashboard (refresh, new, kill)
  - CB_STATUS_*: Status message action buttons (esc, screenshot, recall)
  - CB_RECOVERY_*: Dead window recovery UI (fresh, continue, resume)
  - CB_KEYS_PREFIX: Screenshot control keys (kb:<key_id>:<window>)
"""

# Delimiter between window_id and pane_id in pane-targeted callback data.
# Must not be colon (:) — herdr ids already contain colons (e.g. w2:t1, w2:p1).
CB_PANE_DELIMITER = "|"

# History pagination
CB_HISTORY_PREV = "hp:"  # history page older
CB_HISTORY_NEXT = "hn:"  # history page newer

# Directory browser
CB_DIR_SELECT = "db:sel:"
CB_DIR_UP = "db:up"
CB_DIR_CONFIRM = "db:confirm"
CB_DIR_CANCEL = "db:cancel"
CB_DIR_PAGE = "db:page:"
CB_DIR_FAV = "db:fav:"  # db:fav:<idx> — select a favorite directory
CB_DIR_STAR = "db:star:"  # db:star:<idx> — star/unstar a directory
CB_DIR_HOME = "db:home"  # jump to home directory

# Window picker (bind existing unbound window)
CB_WIN_BIND = "wb:sel:"  # wb:sel:<index>
CB_WIN_NEW = "wb:new"  # proceed to directory browser
CB_WIN_CANCEL = "wb:cancel"

# Screenshot
CB_SCREENSHOT_REFRESH = "ss:ref:"

# Interactive UI (aq: prefix kept for backward compatibility)
CB_ASK_UP = "aq:up:"  # aq:up:<window>
CB_ASK_DOWN = "aq:down:"  # aq:down:<window>
CB_ASK_LEFT = "aq:left:"  # aq:left:<window>
CB_ASK_RIGHT = "aq:right:"  # aq:right:<window>
CB_ASK_ESC = "aq:esc:"  # aq:esc:<window>
CB_ASK_ENTER = "aq:enter:"  # aq:enter:<window>
CB_ASK_SPACE = "aq:spc:"  # aq:spc:<window>
CB_ASK_TAB = "aq:tab:"  # aq:tab:<window>
CB_ASK_REFRESH = "aq:ref:"  # aq:ref:<window>

# Sessions dashboard
CB_SESSIONS_REFRESH = "sess:ref"
CB_SESSIONS_NEW = "sess:new"
CB_SESSIONS_KILL = "sess:kill:"  # sess:kill:<window_id>
CB_SESSIONS_KILL_CONFIRM = "sess:killok:"  # sess:killok:<window_id>

# Status message action buttons
CB_STATUS_ESC = "st:esc:"  # st:esc:<window_id>
CB_STATUS_SCREENSHOT = "st:ss:"  # st:ss:<window_id>
CB_STATUS_RECALL = "st:rc:"  # st:rc:<window_id>:<history_index>
CB_STATUS_LAST_REPLY = "st:lr:"  # st:lr:<window_id>
CB_STATUS_GET_FILE = "st:gf:"  # st:gf:<window_id>

# Recovery UI (dead window)
CB_RECOVERY_FRESH = "rec:f:"  # rec:f:<window_id>
CB_RECOVERY_CONTINUE = "rec:c:"  # rec:c:<window_id>
CB_RECOVERY_RESUME = "rec:r:"  # rec:r:<window_id>
CB_RECOVERY_PICK = "rec:p:"  # rec:p:<index> (resume picker selection)
CB_RECOVERY_BACK = "rec:b:"  # rec:b:<window_id> (back to recovery menu)
CB_RECOVERY_BROWSE = "rec:br:"  # rec:br:<window_id> (browse other projects)
CB_RECOVERY_CANCEL = "rec:x"  # cancel recovery

# Resume command (browse all sessions)
CB_RESUME_PICK = "res:p:"  # res:p:<index> (session selection)
CB_RESUME_PAGE = "res:pg:"  # res:pg:<page> (pagination)
CB_RESUME_CANCEL = "res:x"  # cancel resume browser

# Provider selection (directory browser flow)
CB_PROV_SELECT = "prov:"  # prov:<provider_name>
CB_MODE_SELECT = "mode:"  # mode:<provider_name>:<normal|yolo>

# Worktree picker (directory browser flow — inserted before provider pick
# when the confirmed directory is an eligible git repo)
CB_WT_USE_CURRENT = "wt:cur"  # keep current branch, fall through to provider pick
CB_WT_NEW = "wt:new"  # show confirm/edit screen with suggested branch
CB_WT_CONFIRM = "wt:ok"  # create the worktree, fall through to provider pick
CB_WT_EDIT_NAME = "wt:ed"  # prompt for branch name via text reply

# Workspace picker (directory browser flow — inserted before provider pick on
# backends with native_agent_status=True, e.g. herdr; skipped on tmux).
CB_WS_SELECT = "ws:sel:"  # ws:sel:<index> — select workspace at cached index
CB_WS_SKIP = "ws:skip"  # skip picker, auto-resolve workspace from cwd

# Pane screenshot (from /panes command)
CB_PANE_SCREENSHOT = "pn:ss:"  # pn:ss:<window_id>|<pane_id>

# Pane subscription / rename / lifecycle (Theme 5)
CB_PANE_SUBSCRIBE = "pn:sub:"  # pn:sub:<window_id>|<pane_id>
CB_PANE_UNSUBSCRIBE = "pn:uns:"  # pn:uns:<window_id>|<pane_id>
CB_PANE_RENAME = "pn:rn:"  # pn:rn:<window_id>|<pane_id>
CB_PANE_LIFECYCLE_TOGGLE = "pn:lc:"  # pn:lc:<window_id> — per-window toggle

# Screenshot control keys
CB_KEYS_PREFIX = "kb:"  # kb:<key_id>:<window> or kb:<key_id>:<window_id>|<pane_id>

# Toolbar — single prefix; the suffix encodes "<window_id>:<action_name>".
# The action_name is looked up in the loaded ToolbarConfig.actions pool to
# determine dispatch (key send / text send / builtin handler).
CB_TOOLBAR = "tb:"  # tb:<window_id>:<action_name>

# Sync command
CB_SYNC_FIX = "sync:fix"
CB_SYNC_DISMISS = "sync:x"

# Voice transcription confirm/discard
CB_VOICE = "vc:"  # vc:send:<msg_id> / vc:drop:<msg_id>

# Shell command approval
CB_SHELL_RUN = "sh:run:"  # sh:run:<window_id>
CB_SHELL_EDIT = "sh:edt:"  # sh:edt:<window_id>
CB_SHELL_CANCEL = "sh:x:"  # sh:x:<window_id>
CB_SHELL_CONFIRM_DANGER = "sh:dng:"  # sh:dng:<window_id> (dangerous confirm)

# Live view (auto-refreshing screenshot)
CB_LIVE_START = "lv:go:"  # lv:go:<target> (window_id or window_id:pane_id)
CB_LIVE_STOP = "lv:stop:"  # lv:stop:<target>

# /send command file browser
CB_SEND_FILE = "sf:f:"  # sf:f:<idx> — select file at index
CB_SEND_DIR = "sf:d:"  # sf:d:<idx> — navigate into dir at index
CB_SEND_PAGE = "sf:pg:"  # sf:pg:<page> — pagination
CB_SEND_UP = "sf:up"  # navigate to parent directory
CB_SEND_CANCEL = "sf:x"  # cancel /send browser

# /agent command \u2014 manual provider override picker
CB_AGENT_SET = "ag:set:"  # ag:set:<window_id>:<provider_or_auto>
CB_AGENT_CANCEL = "ag:x:"  # ag:x:<window_id>

# Idle status sentinel (shared between status_polling and message_queue)
IDLE_STATUS_TEXT = "\u2713 Ready"
