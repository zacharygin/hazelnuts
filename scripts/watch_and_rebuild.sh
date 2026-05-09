#!/bin/bash
# Watches era5/era5_temp_*.nc for new/updated files and rebuilds frost CSV.
# Runs until the download process (fetch_era5_frost_historical.py) exits.

REPO="/Users/zacharygin/Documents/oros_prophet - hazelnuts"
LOG="$REPO/logs/rebuild_watch.log"
SCRIPT="$REPO/scripts/fetch_era5_frost_historical.py"
PYTHON="$REPO/.venv/bin/python3"

echo "[$(date)] Watcher started" >> "$LOG"

last_count=0

while pgrep -f "fetch_era5_frost_historical.py" > /dev/null; do
    count=$(ls "$REPO/data/raw/era5/era5_temp_"*.nc 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -ne "$last_count" ]; then
        echo "[$(date)] $count nc files on disk — rebuilding CSV ..." >> "$LOG"
        "$PYTHON" "$SCRIPT" --rebuild-csv-only >> "$LOG" 2>&1
        echo "[$(date)] Rebuild done." >> "$LOG"
        last_count=$count
    fi
    sleep 120
done

# Final rebuild after download completes
echo "[$(date)] Download process ended — final rebuild ..." >> "$LOG"
"$PYTHON" "$SCRIPT" --rebuild-csv-only >> "$LOG" 2>&1
echo "[$(date)] All done." >> "$LOG"
