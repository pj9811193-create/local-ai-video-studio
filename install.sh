#!/data/data/com.termux/files/usr/bin/env bash
# =============================================================
#  LocalAI Video Studio - ONE-COMMAND INSTALLER
#
#  Paste this single line into Termux:
#    curl -fsSL https://raw.githubusercontent.com/pj9811193-create/local-ai-video-studio/main/install.sh | bash
#
#  It installs Termux packages, downloads the app, and launches
#  the self-healing run.sh (which sets up the GPU engine and
#  starts the studio). Fully automated - no questions asked.
# =============================================================
set -e
DEST="$HOME/aivideo"

if [ ! -d "/data/data/com.termux" ]; then
    echo ""
    echo "This installer is meant for Termux on Android."
    echo "Install Termux from https://f-droid.org/packages/com.termux/ first,"
    echo "then paste the install command again."
    exit 1
fi

bold() { printf "\n\033[1;35m%s\033[0m\n" "$1"; }

bold "[1/3] Installing base packages…"
pkg update -y || true
pkg install -y git curl || true

bold "[2/3] Downloading LocalAI Video Studio…"
if [ -d "$DEST/.git" ]; then
    echo "  already installed - updating…"
    git -C "$DEST" pull --ff-only || echo "  (could not update - continuing with local copy)"
else
    rm -rf "$DEST"
    git clone https://github.com/pj9811193-create/local-ai-video-studio.git "$DEST"
fi

bold "[3/3] Launching (auto-setup + start)…"
exec bash "$DEST/run.sh"
