#!/data/data/com.termux/files/usr/bin/env bash
# =============================================================
#  LocalAI Video Studio - fully automated setup (no questions asked)
#
#  Everything here runs 100% offline AFTER this one-time setup.
#
#  Engine options (env vars):
#    ENGINE=webgpu  (default) SD-Turbo on your GPU via Chrome WebGPU.
#                   No compiling, ~2.4 GB download, much faster.
#    ENGINE=cpu     stable-diffusion.cpp on CPU. ~2.3 GB download,
#                   plus a 10-30 min compile. Works on any phone.
#    ENGINE=both    install both engines.
#    ENGINE=demo    no download, no compile (test mode).
#
#  Extra knobs: MODEL=q8|q4|sd15 (cpu engine), SKIP_START=1
# =============================================================
set -e
cd "$(dirname "$0")"

ENGINE="${ENGINE:-webgpu}"
MODEL="${MODEL:-auto}"
SKIP_START="${SKIP_START:-0}"

bold()   { printf "\n\033[1;35m%s\033[0m\n" "$1"; }
info()   { printf "  \033[0;36m>\033[0m %s\n" "$1"; }

dl() {  # resumable download: dl <output> <url>
    [ -s "$1" ] && { info "$1 already downloaded"; return 0; }
    info "downloading $(basename "$1") …"
    curl -L -C - --retry 5 -o "$1.part" "$2" && mv "$1.part" "$1"
}

is_termux() { [ -n "$TERMUX_VERSION" ]; }

# ---------------------------------------------------------------- 1. deps
bold "[1/4] Installing base packages…"
if is_termux; then
    pkg update -y || true
    pkg install -y python ffmpeg git || true
else
    # generic Linux/macOS: check and hint
    command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found - install it for video rendering"
fi
python -m pip install --quiet flask pillow 2>/dev/null || pip install flask pillow

# ---------------------------------------------------------------- 2. webgpu
install_webgpu() {
    bold "[WebGPU] Installing the GPU engine (no compile needed)…"
    mkdir -p webgpu/vendor webgpu/models/sd-turbo/{unet,text_encoder,vae_decoder} webgpu/models/clip-tokenizer

    info "fetching onnxruntime-web (WebGPU build)…"
    if [ ! -f webgpu/vendor/ort.webgpu.min.js ]; then
        curl -sL -o /tmp/ort.tgz \
          https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.19.2.tgz
        tar xzf /tmp/ort.tgz -C webgpu/vendor --strip-components=2 package/dist
        rm -f /tmp/ort.tgz
    else
        info "onnxruntime-web already installed"
    fi

    # SD-Turbo ONNX models (~2.4 GB total)
    dl webgpu/models/sd-turbo/text_encoder/model.onnx \
       "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main/text_encoder/model.onnx"
    dl webgpu/models/sd-turbo/unet/model.onnx \
       "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main/unet/model.onnx"
    dl webgpu/models/sd-turbo/vae_decoder/model.onnx \
       "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main/vae_decoder/model.onnx"

    # CLIP tokenizer (small)
    dl webgpu/models/clip-tokenizer/vocab.json \
       "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main/tokenizer/vocab.json"
    dl webgpu/models/clip-tokenizer/merges.txt \
       "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main/tokenizer/merges.txt"
    info "GPU engine installed. Open the app in Chrome (121+) to use it."
}

# ---------------------------------------------------------------- 3. cpu
install_cpu() {
    bold "[CPU] Building stable-diffusion.cpp (10-30 min, one time)…"
    if [ ! -d stable-diffusion.cpp ]; then
        git clone --recursive --quiet https://github.com/leejet/stable-diffusion.cpp
    fi
    ( cd stable-diffusion.cpp
      mkdir -p build && cd build
      cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null
      JOBS=2   # low-RAM friendly; raise on 6GB+ phones
      cmake --build . --config Release -j"$JOBS" 2>&1 | tail -2 )
    mkdir -p bin
    cp stable-diffusion.cpp/build/bin/sd-cli bin/ 2>/dev/null \
      || cp stable-diffusion.cpp/build/bin/sd bin/

    mkdir -p models
    if [ ! -f models/taesd.safetensors ]; then
        dl models/taesd.safetensors \
           "https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors"
    fi

    if ls models/*.gguf models/*.safetensors 2>/dev/null | grep -qv taesd; then
        info "CPU model already present"
    else
        # auto-pick by RAM: Q4 for phones under ~5 GB
        if [ "$MODEL" = "auto" ]; then
            RAM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 8000000)
            if [ "$RAM_KB" -lt 5000000 ]; then MODEL=q4; else MODEL=q8; fi
        fi
        case "$MODEL" in
          q4)  dl models/sd-turbo-Q4_0.gguf "https://huggingface.co/gpustack/stable-diffusion-v2-1-turbo-GGUF/resolve/main/stable-diffusion-v2-1-turbo-Q4_0.gguf" ;;
          sd15) dl models/v1-5-pruned-emaonly.safetensors "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" ;;
          *)   dl models/sd-turbo-Q8_0.gguf "https://huggingface.co/gpustack/stable-diffusion-v2-1-turbo-GGUF/resolve/main/stable-diffusion-v2-1-turbo-Q8_0.gguf" ;;
        esac
    fi
    info "CPU engine installed."
}

# ---------------------------------------------------------------- run
case "$ENGINE" in
  webgpu) install_webgpu ;;
  cpu)    install_cpu ;;
  both)   install_webgpu; install_cpu ;;
  demo)   info "demo mode - nothing to download" ;;
  *)      echo "unknown ENGINE=$ENGINE (use webgpu|cpu|both|demo)"; exit 1 ;;
esac

bold "[4/4] Setup complete!"
if [ "$SKIP_START" != "1" ]; then
    exec bash run.sh
else
    echo "  Start the app anytime:   bash run.sh"
fi
