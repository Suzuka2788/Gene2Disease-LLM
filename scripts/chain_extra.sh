#!/usr/bin/env bash
# Start the second EF batch once the main batch has finished cleanly.
# Exits without starting it if the main driver stops early (an error, or a closed session).
cd "$(dirname "$0")/.." || exit 1
progress=runs/ef_dev_progress.jsonl
while true; do
  if grep -q '"event": "driver_finished"' "$progress"; then
    break
  fi
  alive=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_ef_dev.py' -and \$_.CommandLine -notmatch 'extra' }).Count")
  if [ "${alive//[$'\r\n ']/}" = "0" ]; then
    echo "main driver is no longer running and did not finish; extra batch not started" >&2
    exit 2
  fi
  sleep 60
done
python scripts/run_ef_dev.py extra >> runs/ef_dev_driver.log 2>&1
