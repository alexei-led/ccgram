#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMUX_SESSION="${CCGRAM_DEV_TMUX_SESSION:-ccgram}"
TMUX_WINDOW="${CCGRAM_DEV_TMUX_WINDOW:-__main__}"
TARGET="${TMUX_SESSION}:${TMUX_WINDOW}"
LOCK_DIR="${PROJECT_DIR}/.ccgram-dev-run.lock.d"

usage() {
	echo "Usage: $0 {start|stop|restart|status}"
	echo "  start   start local dev supervisor in ${TARGET}"
	echo "  stop    stop supervisor loop (Ctrl-\\ in control pane)"
	echo "  restart restart ccgram process (Ctrl-C in control pane)"
	echo "  status  show target pane command and recent logs"
}

runloop() {
	cd "${PROJECT_DIR}"
	acquire_lock() {
		if mkdir "${LOCK_DIR}" 2>/dev/null; then
			echo "$$" >"${LOCK_DIR}/pid"
			trap 'rm -rf "${LOCK_DIR}"' EXIT INT TERM
			return
		fi
		local pid=""
		if [[ -f "${LOCK_DIR}/pid" ]]; then
			pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
		fi
		if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
			echo "[ccgram-dev] supervisor already running (pid ${pid})"
			exit 1
		fi
		rm -rf "${LOCK_DIR}" 2>/dev/null || true
		if mkdir "${LOCK_DIR}" 2>/dev/null; then
			echo "$$" >"${LOCK_DIR}/pid"
			trap 'rm -rf "${LOCK_DIR}"' EXIT INT TERM
			return
		fi
		echo "[ccgram-dev] failed to acquire lock at ${LOCK_DIR}"
		exit 1
	}

	acquire_lock
	ulimit -c 0
	echo "[ccgram-dev] started in ${TARGET}"
	echo "[ccgram-dev] hint: Ctrl-C restarts ccgram"
	echo "[ccgram-dev] hint: Ctrl-\\ stops supervisor loop"
	while true; do
		echo "[ccgram-dev] starting uv run ccgram ($(date '+%H:%M:%S'))"
		set +e
		uv run ccgram
		code=$?
		set -e
		case "${code}" in
		130)
			echo "[ccgram-dev] restart requested"
			sleep 1
			;;
		131)
			echo "[ccgram-dev] stop requested"
			exit 0
			;;
		0)
			echo "[ccgram-dev] exited cleanly; restarting in 1s"
			sleep 1
			;;
		*)
			echo "[ccgram-dev] exited code ${code}; restarting in 1s"
			sleep 1
			;;
		esac
	done
}

ensure_target() {
	if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
		echo "Creating tmux session '${TMUX_SESSION}' with window '${TMUX_WINDOW}'..."
		tmux new-session -d -s "${TMUX_SESSION}" -n "${TMUX_WINDOW}" -c "${PROJECT_DIR}"
		return
	fi
	if ! tmux list-windows -t "${TMUX_SESSION}" -F '#{window_name}' | grep -qx "${TMUX_WINDOW}"; then
		echo "Creating window '${TMUX_WINDOW}' in session '${TMUX_SESSION}'..."
		tmux new-window -t "${TMUX_SESSION}" -n "${TMUX_WINDOW}" -c "${PROJECT_DIR}"
	fi
}

pane_command() {
	tmux list-panes -t "${TARGET}" -F '#{pane_current_command}' 2>/dev/null | head -n 1
}

status() {
	ensure_target
	local cmd
	cmd="$(pane_command || true)"
	echo "Target: ${TARGET}"
	echo "Current command: ${cmd:-unknown}"
	echo "Recent output:"
	echo "----------------------------------------"
	tmux capture-pane -t "${TARGET}" -p | tail -20
	echo "----------------------------------------"
}

start() {
	ensure_target
	local cmd
	cmd="$(pane_command || true)"
	if [[ -n "${cmd}" && "${cmd}" != "zsh" && "${cmd}" != "bash" && "${cmd}" != "sh" && "${cmd}" != "fish" ]]; then
		echo "Target pane ${TARGET} is busy (current command: ${cmd})."
		echo "Use '$0 stop' or clear the pane before starting."
		exit 1
	fi

	echo "Starting local dev supervisor in ${TARGET}..."
	tmux send-keys -t "${TARGET}" "bash '${PROJECT_DIR}/scripts/restart.sh' __runloop" Enter
	sleep 1
	status
}

stop() {
	ensure_target
	echo "Stopping supervisor in ${TARGET} (Ctrl-\\)..."
	tmux send-keys -t "${TARGET}" C-\\
	sleep 1
	status
}

restart() {
	ensure_target
	echo "Restarting ccgram in ${TARGET} (Ctrl-C)..."
	tmux send-keys -t "${TARGET}" C-c
	sleep 1
	status
}

case "${1:-}" in
__runloop) runloop ;;
start) start ;;
stop) stop ;;
restart) restart ;;
status) status ;;
*)
	usage
	exit 2
	;;
esac
