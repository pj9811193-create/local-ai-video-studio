# LocalAI Video Studio 🎬

**Website:** https://pj9811193-create.github.io/local-ai-video-studio/

A **fully offline, free, unlimited** text-to-video generator that runs
entirely on your Android phone — no internet, no accounts, no watermarks,
no GPU needed. You type a prompt, an on-device AI model paints scene
images, and the app turns them into a cinematic MP4 with camera motion
and crossfades.

```
your prompt ──> stable-diffusion.cpp (AI, on your phone) ──> keyframes
                                                            │
final MP4  <── ffmpeg (zoom / pan / crossfade)  <───────────┘
```

## What you need

- An Android phone with **at least 4 GB free storage** (for the AI model)
  and ideally 4 GB+ RAM. 3 GB RAM phones work at low resolution.
- **Termux** — get it from **F-Droid** (https://f-droid.org/packages/com.termux/),
  NOT the Play Store (the Play Store version is outdated and broken).
- Wi-Fi for the one-time download of the AI model (~2.3 GB). After that,
  everything runs offline forever.

## Install (one time)

1. Open Termux and run:
   ```bash
   git clone https://github.com/pj9811193-create/local-ai-video-studio.git ~/aivideo
   cd ~/aivideo
   bash setup.sh
   ```
2. `setup.sh` will:
   - install python, ffmpeg and build tools,
   - compile the offline AI engine (stable-diffusion.cpp) — takes
     **10–30 minutes**, plug in your charger,
   - download the AI model you pick (SD-Turbo Q8, ~2.3 GB, recommended).

If any download is interrupted, just run `bash setup.sh` again —
downloads resume from where they stopped.

## Use it

```bash
cd ~/aivideo
bash start.sh
```

Then open **http://localhost:8080** in Chrome (Termux usually opens it
automatically). Type a prompt like:

> a peaceful waterfall in a magical forest, fireflies at dusk, cinematic

pick the number of scenes and camera motion, and hit **Generate**.
Your videos are listed under “Saved videos” and stored in
`~/aivideo/outputs/` — download them from the browser to your gallery.

## Realistic expectations (please read)

This is real AI running on a real phone, so it is **slow but truly
unlimited and private**:

| Phone class | One 384×384 image (SD-Turbo) |
|---|---|
| Flagship (SD 8-series, 8 GB RAM) | ~1–3 min |
| Mid-range (Helio/Snapdragon 6xx) | ~5–15 min |
| Low-end (3 GB RAM, 320×384 only) | ~15–45 min |

A 4-scene video = 4 images + ~30 s of ffmpeg rendering. Keep the phone
**plugged in**, the app open, and the screen on while it works.

What this app is *not*: it is not a movie-model like Sora — those need a
desktop GPU with gigabytes of VRAM and cannot run on a phone. It creates
**AI-generated cinematic scene videos**: a few AI images per video,
animated with camera motion and blended together. That is the honest
maximum for fully-offline generation on low-end hardware.

## Tips for low-end phones

- Start with **3 scenes @ 320 resolution**, then work your way up.
- Use the **Q4_0 model** (option 2 in setup) if the app gets killed.
- Close other apps first; Android kills heavy background processes.
- The first image of a run is slower (model loading); later ones are faster
  only within the same generation.
- If generation crashes the server: lower resolution, use Q4_0, and see below.

## If Android kills the process (Android 12+ “phantom process” fix)

Android 12+ may silently kill long jobs. If your generations die mid-way,
run this once from a PC with ADB, or from Termux:

```bash
adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent; /system/bin/device_config put activity_manager max_phantom_processes 2147483647"
```

(On a PC: enable USB debugging, connect the phone, then run the same
command without `adb shell` quoting differences.)

## Troubleshooting

- **`cmake: command not found`** → run `pkg install cmake` and re-run setup.
- **Build killed** → re-run `bash setup.sh`; it resumes. Or edit `JOBS=2`
  to `JOBS=1` in setup.sh and retry.
- **“demo mode · engine not built”** → the compile step failed; check the
  messages shown during setup.
- **“demo mode · no AI model yet”** → re-run `bash setup.sh` and pick a
  model download.
- **Out of memory / Termux crashes** → use the Q4_0 model, 320 resolution,
  and 3 scenes.

## Where things live

| Path | What |
|---|---|
| `~/aivideo/models/` | AI models (safe to delete to free space) |
| `~/aivideo/outputs/` | your finished MP4 videos |
| `~/aivideo/work/` | temporary files (auto-cleaned) |

To completely uninstall, just delete the `aivideo` folder and
`pkg uninstall` the installed packages.

## Credits

Built on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(SD-Turbo model by Stability AI), TAESD by madebyollin, and ffmpeg.
Model licenses: SD-Turbo is a non-commercial research model; SD 1.5 is
CreativeML OpenRAIL-M. Check the license before commercial use.
