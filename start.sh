#!/data/data/com.termux/files/usr/bin/env bash
# Start LocalAI Video Studio (safe to run on any Linux/macOS too)
cd "$(dirname "$0")"

# Keep the CPU awake while generating (Termux only, harmless elsewhere)
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock

PORT="${PORT:-8080}" python server.py &
SERVER_PID=$!

sleep 2
URL="http://localhost:$PORT"
echo ""
echo "  Video studio running at: $URL"
command -v termux-open-url >/dev/null 2>&1 && termux-open-url "$URL" || true

wait $SERVER_PID
