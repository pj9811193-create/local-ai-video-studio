#!/data/data/com.termux/files/usr/bin/env bash
# =============================================================
#  LocalAI Video Studio - the ONLY command you need.
#
#  Self-healing: installs anything that is missing, then starts
#  the app. Safe to run over and over.
#
#    bash run.sh                 # default: GPU (WebGPU) engine
#    ENGINE=cpu bash run.sh      # install/use the CPU engine
# =============================================================
cd "$(dirname "$0")"

info() { printf "  \033[0;36m>\033[0m %s\n" "$1"; }
is_termux() { [ -n "$TERMUX_VERSION" ]; }

has_webgpu() {
    [ -f webgpu/vendor/ort.webgpu.min.js ] \
      && [ -f webgpu/models/sd-turbo/unet/model.onnx ] \
      && [ -f webgpu/models/sd-turbo/text_encoder/model.onnx ] \
      && [ -f webgpu/models/sd-turbo/vae_decoder/model.onnx ] \
      && [ -f webgpu/models/clip-tokenizer/vocab.json ]
}

has_cpu() {
    [ -x bin/sd-cli ] || [ -x bin/sd ] || [ -x stable-diffusion.cpp/build/bin/sd-cli ]
    ls models/*.gguf models/*.safetensors 2>/dev/null | grep -qv taesd
}

# --- self-heal: missing dependencies? -------------------------
if is_termux; then
    command -v python  >/dev/null || pkg install -y python  > /dev/null 2>&1 || true
    command -v ffmpeg  >/dev/null || pkg install -y ffmpeg  > /dev/null 2>&1 || true
    command -v git     >/dev/null || pkg install -y git     > /dev/null 2>&1 || true
fi
python -c "import flask, PIL" 2>/dev/null \
  || pip install --quiet flask pillow \
  || python -m pip install --quiet flask pillow

# --- self-heal: no engine installed at all? --------------------
if ! has_webgpu && ! has_cpu; then
    echo ""
    info "first run: setting up the GPU engine (~2.4 GB, one time)…"
    info "(for the CPU engine instead:  ENGINE=cpu bash run.sh)"
    echo ""
    ENGINE="${ENGINE:-webgpu}" SKIP_START=1 bash setup.sh
fi

# --- keep the CPU model choice sane if the user asked for cpu ---
if [ "${ENGINE:-}" = "cpu" ] && ! has_cpu; then
    ENGINE=cpu SKIP_START=1 bash setup.sh
fi

# --- start ------------------------------------------------------
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock

PORT="${PORT:-8080}" python server.py &
SERVER_PID=$!

sleep 2
URL="http://localhost:$PORT"
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │   LocalAI Video Studio is running           │"
echo "  │   → $URL"
echo "  └─────────────────────────────────────────────┘"
command -v termux-open-url >/dev/null 2>&1 && termux-open-url "$URL" || true

wait $SERVER_PID
