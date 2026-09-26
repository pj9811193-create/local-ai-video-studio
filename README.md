# LocalAI Video Studio 🎬

**Website:** https://pj9811193-create.github.io/local-ai-video-studio/

A **fully offline, free, unlimited** text-to-video generator that runs
entirely on your Android phone — no internet after setup, no accounts,
no watermarks. You type a prompt, an AI model paints scene images, and the
app turns them into a cinematic MP4 with camera motion and crossfades.

```
your prompt ──> AI engine (GPU WebGPU or CPU, on your phone) ──> keyframes
                                                                  │
final MP4  <──── ffmpeg (zoom / pan / crossfade)  <───────────────┘
```

## Two engines

| | GPU · WebGPU (default) | CPU · stable-diffusion.cpp |
|---|---|---|
| How | SD-Turbo runs **in Chrome, on your GPU**, via WebGPU | SD-Turbo runs on CPU via stable-diffusion.cpp |
| Install | download only, **no compiling** | compiles 10–30 min |
| Speed | fast on supported phones | minutes per image |
| Needs | Chrome 121+ (Android 12+) with WebGPU, decent GPU | any phone |

The app auto-picks the best available engine and falls back gracefully
(GPU → CPU → demo mode).

## Install & run — one command

Install **Termux from F-Droid** (https://f-droid.org/packages/com.termux/,
NOT the Play Store), then:

```bash
git clone https://github.com/pj9811193-create/local-ai-video-studio.git ~/aivideo
cd ~/aivideo && bash run.sh
```

`run.sh` is **self-healing**: it installs whatever is missing (packages,
models) and then starts the app — first run downloads the GPU engine
(~2.4 GB, one time). The browser opens at http://localhost:8080 and you
can disconnect from the internet forever.

To choose the CPU engine instead: `ENGINE=cpu bash run.sh`
(adds compiling + a ~2.3 GB model; also auto-picks Q4/Q8 by your RAM).

Everything is automated — no questions asked. Optional knobs:

```bash
ENGINE=webgpu bash setup.sh   # default: GPU engine, no compile
ENGINE=cpu     bash setup.sh  # CPU engine (MODEL=q8|q4|sd15 to override)
ENGINE=both    bash setup.sh  # both engines
ENGINE=demo    bash setup.sh  # no download at all (test mode)
SKIP_START=1  bash setup.sh  # set up but don't launch
```

## Use it

```bash
cd ~/aivideo && bash run.sh
```

Open **http://localhost:8080** in Chrome, type a prompt like:

> a peaceful waterfall in a magical forest, fireflies at dusk, cinematic

pick scenes, camera motion and engine, hit **Generate**. Videos are
stored in `~/aivideo/outputs/` and listed under “Saved videos”.

## Realistic expectations (please read)

Real AI on real hardware — **slow but truly unlimited and private**:

| Phone class | One 384×384 image, CPU engine | GPU engine |
|---|---|---|
| Flagship (SD 8-series, 8 GB RAM) | ~1–3 min | seconds |
| Mid-range (Helio/Snapdragon 6xx) | ~5–15 min | ~10–60 s |
| Low-end (3 GB RAM) | ~15–45 min | may not support WebGPU — use CPU |

Keep the phone **plugged in** while it works.

What this app is *not*: not a movie-model like Sora (those need a desktop
GPU). It creates **AI-generated cinematic scene videos** — real AI images
animated with camera motion and crossfades. That is the honest maximum for
fully-offline, unlimited generation on low-end hardware.

## Troubleshooting

- **“GPU · browser lacks WebGPU”** → update Chrome (121+), use Chrome
  (not Firefox/Samsung Internet), and make sure the phone runs Android 12+.
- **GPU engine fails mid-generation** → the app suggests switching to CPU;
  that always works, just slower.
- **`cmake: command not found`** (CPU engine) → `pkg install cmake`, re-run.
- **Build killed** (CPU engine) → re-run `ENGINE=cpu bash run.sh`; it resumes.
  Or edit `JOBS=2` → `JOBS=1` in setup.sh.
- **Out of memory / Termux crashes** → use Q4_0 model (`MODEL=q4 bash setup.sh`),
  320 resolution, 3 scenes.
- **Generation dies mid-way on Android 12+** → Android kills long jobs; fix
  once via ADB:
  `adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent; /system/bin/device_config put activity_manager max_phantom_processes 2147483647"`

## Where things live

| Path | What |
|---|---|
| `~/aivideo/webgpu/` | GPU engine (onnx models + onnxruntime-web, ~2.4 GB) |
| `~/aivideo/models/` | CPU AI models |
| `~/aivideo/outputs/` | your finished MP4 videos |
| `~/aivideo/work/` | temporary files (auto-cleaned) |

To uninstall, delete the `aivideo` folder and `pkg uninstall` the packages.

## Development

`bash scripts/ci_test.sh` runs the full pipeline test locally (demo mode).
GitHub Actions runs it automatically on every push (see `.github/workflows/ci.yml`).

## Credits

GPU engine: [onnxruntime-web](https://onnxruntime.ai/) WebGPU example by
Microsoft, SD-Turbo ONNX export by schmuell. CPU engine:
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(SD-Turbo by Stability AI), TAESD by madebyollin, ffmpeg. MIT code
license; SD-Turbo model is non-commercial research use.
