#!/data/data/com.termux/files/usr/bin/env bash
# =============================================================
#  LocalAI Video Studio - one-time setup for Termux (Android)
#  Everything downloaded here runs 100% offline afterwards.
# =============================================================
set -e
cd "$(dirname "$0")"

bold() { printf "\n\033[1;35m%s\033[0m\n" "$1"; }

bold "[1/5] Installing base packages (python, ffmpeg, build tools)..."
pkg update -y
pkg install -y python git cmake clang ffmpeg
python -m pip install --upgrade pip 2>/dev/null || true
pip install flask pillow

bold "[2/5] Building stable-diffusion.cpp (the offline AI engine)..."
# This compiles from source - takes 10-30 minutes on a phone.
# If the build is killed, run setup.sh again: it resumes where it stopped.
if [ ! -d stable-diffusion.cpp ]; then
    git clone --recursive https://github.com/leejet/stable-diffusion.cpp
fi
cd stable-diffusion.cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null
JOBS=2   # low RAM friendly; raise to 4 on phones with 6GB+ RAM
cmake --build . --config Release -j"$JOBS" 2>&1 | tail -3
cd ../..
mkdir -p bin
if [ -f stable-diffusion.cpp/build/bin/sd-cli ]; then
    cp stable-diffusion.cpp/build/bin/sd-cli bin/
elif [ -f stable-diffusion.cpp/build/bin/sd ]; then
    cp stable-diffusion.cpp/build/bin/sd bin/
else
    echo "ERROR: build finished but no binary found - see messages above"; exit 1
fi
echo "AI engine built OK."

bold "[3/5] Downloading TAESD (tiny fast decoder, ~10 MB)..."
mkdir -p models
if [ ! -f models/taesd.safetensors ]; then
    curl -L -C - --retry 5 -o models/taesd.safetensors \
      "https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors"
fi

bold "[4/5] Choose your AI model (one-time download, works offline forever)..."
echo ""
echo "  1) SD-Turbo Q8_0  GGUF   2.3 GB   RECOMMENDED - fast, best quality"
echo "  2) SD-Turbo Q4_0  GGUF   2.2 GB   for very weak phones"
echo "  3) SD 1.5         safet.  4.3 GB   most versatile, but slower (20 steps)"
echo "  4) Skip - stay in demo mode (no download)"
echo ""
read -r -p "Enter 1/2/3/4 [default 1]: " choice
choice=${choice:-1}

dl() {  # resumable download with progress
    curl -L -C - --retry 5 -o "$1" "$2"
}

case "$choice" in
  1) dl models/sd-turbo-Q8_0.gguf \
       "https://huggingface.co/gpustack/stable-diffusion-v2-1-turbo-GGUF/resolve/main/stable-diffusion-v2-1-turbo-Q8_0.gguf" ;;
  2) dl models/sd-turbo-Q4_0.gguf \
       "https://huggingface.co/gpustack/stable-diffusion-v2-1-turbo-GGUF/resolve/main/stable-diffusion-v2-1-turbo-Q4_0.gguf" ;;
  3) dl models/v1-5-pruned-emaonly.safetensors \
       "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" ;;
  4) echo "OK - demo mode. You can download a model later by re-running setup.sh." ;;
  *) echo "Invalid choice, skipping (demo mode)." ;;
esac

bold "[5/5] Done!"
echo ""
echo "  Start the app:    bash start.sh"
echo "  Then open:        http://localhost:8080"
echo ""
