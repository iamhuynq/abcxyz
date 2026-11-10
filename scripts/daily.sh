#!/usr/bin/env bash
# Daily pipeline: fireant update -> fireant validate -> quant build -> quant daily.
# Safe to trigger many times a day: it exits early when today's run already succeeded, when another run
# holds the lock, or when the network/API is down (exit 10: retry at the next trigger).
# Exit codes: 0 done (or nothing to do), 3 token expired, 7 latest session incomplete, 8 validation errors,
# 9 refused (another run holds the lock; nothing done), 10 offline, other = failure.
set -u
export TZ=Asia/Ho_Chi_Minh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA="${DATA_DIR:-data}"
FIREANT="${FIREANT_CMD:-uv run fireant}"
QUANT="${QUANT_CMD:-uv run quant}"
LOCK="$DATA/.daily.lock"
MARK="$DATA/.daily_done"
mkdir -p "$DATA/logs"
LOG="$DATA/logs/daily-$(date +%F).log"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
notify() {
  log "NOTIFY: $1"
  if [ -n "${NOTIFY_CMD:-}" ]; then $NOTIFY_CMD "$1"; return; fi
  osascript -e "display notification \"$1\" with title \"Stock pipeline\"" >/dev/null 2>&1 || true
}

# The session this run is responsible for: today on a weekday after 18:00, else the previous weekday.
target_day() {
  if [ -n "${TARGET_DAY:-}" ]; then echo "$TARGET_DAY"; return; fi
  /usr/bin/python3 - <<'PY'
from datetime import datetime, timedelta
now = datetime.now()
d = now.date() if (now.weekday() < 5 and now.hour >= 18) else now.date() - timedelta(days=1)
while d.weekday() >= 5:
    d -= timedelta(days=1)
print(d.isoformat())
PY
}

# Date and file-time helpers in Python: BSD date/stat flags (macOS) differ from GNU coreutils (Linux, CI).
days_before() { /usr/bin/python3 -c 'import sys, datetime as d; print(d.date.fromisoformat(sys.argv[1]) - d.timedelta(days=int(sys.argv[2])))' "$1" "$2"; }
iso_weekday() { /usr/bin/python3 -c 'import sys, datetime as d; print(d.date.fromisoformat(sys.argv[1]).isoweekday())' "$1"; }
mtime() { /usr/bin/python3 -c 'import os, sys; print(int(os.path.getmtime(sys.argv[1])))' "$1" 2>/dev/null || date +%s; }

TARGET="$(target_day)"
if [ -f "$MARK" ] && [[ ! "$(cat "$MARK")" < "$TARGET" ]]; then
  echo "Already done for $TARGET."; exit 0
fi

# Lock: mkdir is atomic. A lock older than 3 hours or owned by a dead process is stale.
if ! mkdir "$LOCK" 2>/dev/null; then
  pid="$(cat "$LOCK/pid" 2>/dev/null || echo 0)"
  age=$(( $(date +%s) - $(mtime "$LOCK") ))
  if kill -0 "$pid" 2>/dev/null && [ "$age" -lt 10800 ]; then
    echo "Another run (pid $pid) is in progress; nothing done."; exit 9
  fi
  log "Removing stale lock (pid $pid, age ${age}s)"
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

log "Start (target session $TARGET)"
$FIREANT update >>"$LOG" 2>&1; rc=$?
case $rc in
  0) ;;
  10) notify "Offline or API down; nothing changed. Will retry at the next trigger."; exit 10 ;;
  3)  notify "FireAnt token expired: update FIREANT_TOKEN in .env."; exit 3 ;;
  *)  notify "fireant update failed (exit $rc); see $LOG"; exit "$rc" ;;
esac
# Weekly jobs: on a Friday target, or when the last successful weekly update is 7+ days old (missed Fridays).
WEEKLY_MARK="$DATA/.weekly_done"
WEEK_AGO="$(days_before "$TARGET" 7)"
if [ "$(iso_weekday "$TARGET")" = "5" ] || [ ! -f "$WEEKLY_MARK" ] || [[ ! "$(cat "$WEEKLY_MARK")" > "$WEEK_AGO" ]]; then
  if $FIREANT update --jobs report_marks,fundamental >>"$LOG" 2>&1; then
    echo "$TARGET" > "$WEEKLY_MARK"
  else
    log "Weekly report_marks/fundamental update had failures (not blocking; retried at the next run)"
  fi
fi
# Error-level findings (the latest sessions look broken) stop the run before the build; warn/info do not.
$FIREANT validate >>"$LOG" 2>&1; rc=$?
if [ "$rc" -ne 0 ]; then
  notify "Validation found error-level problems in the latest sessions; build stopped. See data/reports and $LOG"; exit 8
fi
$QUANT build >>"$LOG" 2>&1 || { rc=$?; notify "quant build failed (exit $rc); see $LOG"; exit "$rc"; }
$QUANT daily >>"$LOG" 2>&1; rc=$?
case $rc in
  0) echo "$TARGET" > "$MARK"; notify "Done for $TARGET: report in data/reports/daily/"; exit 0 ;;
  7) notify "Latest session incomplete in the source; will retry at the next trigger."; exit 7 ;;
  *) notify "quant daily failed (exit $rc); see $LOG"; exit "$rc" ;;
esac
