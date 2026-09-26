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

## Engines

| | GPU · WebGPU (default) | Custom · own tiny model | CPU · stable-diffusion.cpp |
|---|---|---|---|
| How | SD-Turbo in Chrome via WebGPU | our own ~0.9M-param model, trained from scratch | SD-Turbo on CPU |
| Download | ~1.8–2.4 GB | **~4 MB** | ~2.3 GB + compile |
| Quality | best | simple stylized scenes (12 categories) | good |
| Needs | Chrome 121+, decent GPU | any modern browser | any phone, patience |

The app auto-picks the best available engine and falls back gracefully
(GPU → own model → CPU → demo mode).

## Install & run — one command

Install **Termux from F-Droid** (https://f-droid.org/packages/com.termux/,
NOT the Play Store), open it, and paste this single line:

```bash
curl -fsSL https://raw.githubusercontent.com/pj9811193-create/local-ai-video-studio/main/install.sh | bash
```

That's it. It installs the packages, downloads the app, sets up the GPU
engine (~2.4 GB, one time) and starts the studio in your browser.

To choose the CPU engine instead: `ENGINE=cpu bash run.sh`
(adds compiling + a ~2.3 GB model; also auto-picks Q4/Q8 by your RAM).

Low on storage? Use the int8-compressed GPU engine: `ENGINE=webgpu8 bash run.sh`
(~1.8 GB instead of ~2.4 GB — same pipeline, built by `scripts/make_int8.py`
and hosted on this repo's [releases](https://github.com/pj9811193-create/local-ai-video-studio/releases/tag/webgpu-int8);
the fp16 engine stays the quality reference).

**Our own tiny AI model**: `ENGINE=custom bash run.sh` installs a ~4 MB model
that we trained completely from scratch (our architecture + our procedural
training set, `scripts/train_custom_model.py` — no third-party weights).
It paints simple stylized scenes (12 categories picked from your prompt:
sunset, ocean, forest, mountains, night, city, desert, snow, meadow, space,
fire, abstract) at low detail and runs on any phone. It is a real own model,
not SD-Turbo quality — the honest trade-off for 4 MB.

After install, this is the only command you ever need again:

```bash
bash ~/aivideo/run.sh
```

Everything is automated — no questions asked. Optional knobs:

```bash
ENGINE=webgpu bash setup.sh   # default: GPU engine, no compile (~2.4 GB)
ENGINE=webgpu8 bash setup.sh  # int8-compressed GPU engine (~1.8 GB)
ENGINE=custom bash setup.sh  # our own tiny model (~4 MB, stylized scenes)
ENGINE=cpu     bash setup.sh  # CPU engine (MODEL=q8|q4|sd15 to override)
ENGINE=both    bash setup.sh  # both engines
ENGINE=all     bash setup.sh  # GPU + own tiny model + CPU
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
| `~/aivideo/webgpu/` | GPU engine (onnx models + onnxruntime-web, ~1.8–2.4 GB) |
| `~/aivideo/models/` | CPU AI models |
| `~/aivideo/outputs/` | your finished MP4 videos |
| `~/aivideo/work/` | temporary files (auto-cleaned) |

To uninstall, delete the `aivideo` folder and `pkg uninstall` the packages.

## Development

`bash scripts/ci_test.sh` runs the full pipeline test locally (demo mode).
GitHub Actions runs it automatically on every push (see `.github/workflows/ci.yml`).

Own-model training: `.github/workflows/train-custom-model.yml` (manual) retrains
`scripts/train_custom_model.py` from scratch and publishes to the
[custom-model release](https://github.com/pj9811193-create/local-ai-video-studio/releases/tag/custom-model).

## Credits

GPU engine: [onnxruntime-web](https://onnxruntime.ai/) WebGPU example by
Microsoft, SD-Turbo ONNX export by schmuell. CPU engine:
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(SD-Turbo by Stability AI), TAESD by madebyollin, ffmpeg. MIT code
license; SD-Turbo model is non-commercial research use.
