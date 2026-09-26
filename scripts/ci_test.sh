#!/usr/bin/env bash
# End-to-end test of the LocalAI Video Studio pipeline in demo mode.
# Starts the server, asks for a 3-scene video, and verifies the MP4.
# Used by CI; safe to run anywhere with python+ffmpeg.
set -e
cd "$(dirname "$0")/.."

PORT="${TEST_PORT:-8123}"
rm -rf outputs work
mkdir -p outputs work

PORT=$PORT python server.py > server_test.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

# wait for the server
for i in $(seq 1 30); do
    sleep 1
    curl -sf "http://localhost:$PORT/api/status" > /dev/null && break
done

echo "→ server is up:"
curl -s "http://localhost:$PORT/api/status"; echo

echo "→ starting demo generation…"
JOB=$(curl -s -X POST "http://localhost:$PORT/api/generate" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a peaceful waterfall in a magical forest, cinematic","frames":3,"sec":3.0,"resolution":320,"motion":"auto","evolution":true}' \
    | sed 's/.*"job_id":"\([a-f0-9]*\)".*/\1/')
echo "  job: $JOB"

for i in $(seq 1 60); do
    sleep 2
    S=$(curl -s "http://localhost:$PORT/api/job/$JOB")
    ST=$(echo "$S" | sed 's/.*"status":"\([a-z]*\)".*/\1/')
    echo "  [$i] $ST"
    if [ "$ST" = "done" ] || [ "$ST" = "error" ]; then break; fi
done

echo "$S" | grep -q '"status":"done"' || { echo "FAILED: $S"; cat server_test.log; exit 1; }

echo "→ downloading result…"
curl -sf -o test_video.mp4 "http://localhost:$PORT/api/video/$JOB.mp4"
SIZE=$(stat -c%s test_video.mp4)
[ "$SIZE" -gt 10000 ] || { echo "FAILED: video too small ($SIZE bytes)"; exit 1; }
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 test_video.mp4 2>/dev/null || ffmpeg -i test_video.mp4 2>&1 | grep -o 'Duration: [0-9:.]*' | head -1)
echo "  mp4: $SIZE bytes, $DUR"

echo ""
echo "✓ PIPELINE TEST PASSED"
rm -f test_video.mp4
