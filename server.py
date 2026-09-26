#!/usr/bin/env python3
"""
LocalAI Video Studio - fully offline text-to-video generator for Android (Termux).

How it works:
  1. Your prompt is sent to an AI engine running 100% on your phone
     (fully offline): either stable-diffusion.cpp on CPU, or SD-Turbo in
     the browser on your GPU via WebGPU (onnxruntime-web).
     The engine generates a few AI keyframe images.
  2. ffmpeg animates each keyframe with cinematic camera motion
     (zoom / pan) and blends them together with crossfades.
  3. The finished MP4 is served to you in the browser.

No internet connection is needed after the one-time setup.
No accounts, no limits, no watermarks - everything runs on your device.

If no AI model is installed yet, the app runs in DEMO mode (test frames)
so you can verify the whole pipeline before downloading a model.
"""

import base64
import colorsys
import glob
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_file, abort

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
WEBGPU_DIR = os.path.join(APP_DIR, "webgpu")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
WORK_DIR = os.path.join(APP_DIR, "work")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

PORT = int(os.environ.get("PORT", "8080"))
FPS = 24

MODEL_EXTS = (".gguf", ".safetensors", ".ckpt", ".pt")

# ---------------------------------------------------------------- helpers

def find_sd_bin():
    """Locate the stable-diffusion.cpp binary."""
    candidates = [
        os.path.join(APP_DIR, "bin", "sd-cli"),
        os.path.join(APP_DIR, "bin", "sd"),
        os.path.join(APP_DIR, "stable-diffusion.cpp", "build", "bin", "sd-cli"),
        os.path.join(APP_DIR, "stable-diffusion.cpp", "build", "bin", "sd"),
        shutil.which("sd-cli"),
        shutil.which("sd"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def find_model():
    """Pick the first model file in ./models (prefer GGUF - already quantized)."""
    files = []
    for ext in MODEL_EXTS:
        files.extend(glob.glob(os.path.join(MODELS_DIR, "*" + ext)))
    ggufs = [f for f in files if f.endswith(".gguf")]
    for f in (ggufs + files):
        if os.path.basename(f) != "taesd.safetensors":
            return f
    return None


def find_taesd():
    p = os.path.join(MODELS_DIR, "taesd.safetensors")
    return p if os.path.isfile(p) else None


def find_webgpu_files():
    """True if the WebGPU engine assets (onnx models + ort vendor) are installed."""
    needed = [
        os.path.join(WEBGPU_DIR, "vendor", "ort.webgpu.min.js"),
        os.path.join(WEBGPU_DIR, "models", "sd-turbo", "unet", "model.onnx"),
        os.path.join(WEBGPU_DIR, "models", "sd-turbo", "text_encoder", "model.onnx"),
        os.path.join(WEBGPU_DIR, "models", "sd-turbo", "vae_decoder", "model.onnx"),
        os.path.join(WEBGPU_DIR, "models", "clip-tokenizer", "vocab.json"),
        os.path.join(WEBGPU_DIR, "models", "clip-tokenizer", "merges.txt"),
    ]
    return all(os.path.isfile(p) for p in needed)


SD_BIN = find_sd_bin()
MODEL = find_model()
TAESD = find_taesd()
AI_MODE = bool(SD_BIN and MODEL)
WEBGPU_FILES = find_webgpu_files()


def model_profile(path):
    """Return sensible sampler settings for the installed model."""
    name = os.path.basename(path).lower()
    if "turbo" in name:
        # distilled models: very few steps, no CFG (negative prompts ignored)
        return {"steps": 4, "cfg": 1.0, "sampler": "euler"}
    return {"steps": 20, "cfg": 7.0, "sampler": "euler a"}


# ---------------------------------------------------------------- job store

JOBS = {}
JOBS_LOCK = threading.Lock()
BUSY_LOCK = threading.Lock()   # a phone can only handle one generation at a time


def job_snapshot(j):
    return {
        "id": j["id"],
        "status": j["status"],
        "stage": j["stage"],
        "progress": round(j["progress"], 1),
        "frames_done": j["frames_done"],
        "total_frames": j["total_frames"],
        "elapsed": round(time.time() - j["t0"], 1),
        "error": j["error"],
        "video_url": ("/api/video/%s.mp4" % j["id"]) if j["status"] == "done" else None,
        "thumbs": ["/api/frame/%s/%d.png" % (j["id"], i) for i in range(j["frames_done"])],
        "log": j["log"][-6:],
    }


def make_demo_frame(path, w, h, idx, total, prompt):
    """Placeholder keyframe used in DEMO mode (no AI model installed)."""
    from PIL import Image, ImageDraw, ImageFilter

    hue = (idx * 36 + random.randint(-8, 8)) % 360
    c_top = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 0.65, 0.30))
    c_bot = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(((hue + 60) % 360) / 360, 0.55, 0.08))
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        d.line([(0, y), (w, y)],
               fill=tuple(int(a + (b - a) * t) for a, b in zip(c_top, c_bot)))
    # soft glowing circles
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    rng = random.Random(idx * 977 + 5)
    for _ in range(6):
        r = rng.randint(w // 10, w // 3)
        x, y = rng.randint(0, w), rng.randint(0, h)
        cc = tuple(int(c * 255) for c in
                   colorsys.hsv_to_rgb(rng.random(), 0.8, rng.uniform(0.4, 0.9)))
        gd.ellipse([x - r, y - r, x + r, y + r], fill=cc)
    glow = glow.filter(ImageFilter.GaussianBlur(w // 12))
    img = Image.blend(img, glow, 0.5)
    d = ImageDraw.Draw(img)
    text = prompt if len(prompt) <= 42 else prompt[:40] + ".."
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(28)
        small = ImageFont.load_default(18)
    except Exception:
        font = small = None
    d.text((24, h // 2 - 40), "DEMO MODE", fill=(255, 255, 255), font=font)
    d.text((24, h // 2), text, fill=(235, 235, 235), font=font)
    d.text((24, h - 46), "frame %d/%d - install an AI model for real images"
           % (idx + 1, total), fill=(200, 200, 200), font=small)
    img.save(path)


SHOT_VARIATIONS = [
    "wide establishing shot",
    "medium shot",
    "close-up detail shot",
    "different angle, dramatic lighting",
    "golden hour lighting",
    "slightly different composition",
]


def build_sd_cmd(model, prompt, negative, out_png, w, h, seed, prof, threads):
    cmd = [
        SD_BIN,
        "-m", model,
        "-p", prompt,
        "-o", out_png,
        "--steps", str(prof["steps"]),
        "--cfg-scale", str(prof["cfg"]),
        "--sampling-method", prof["sampler"],
        "--seed", str(seed),
        "-W", str(w), "-H", str(h),
        "-t", str(threads),
    ]
    if negative:
        cmd += ["-n", negative]
    if TAESD:
        cmd += ["--taesd", TAESD]
    if not model.endswith(".gguf"):
        # runtime quantization for fat .safetensors/.ckpt files (saves RAM)
        cmd += ["--type", "q8_0"]
    return cmd


def run_sd(cmds, out_png, log):
    """Try command variants until one works (sd.cpp CLI differs across versions)."""
    last_err = ""
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
            if r.returncode == 0 and os.path.isfile(out_png):
                return True, ""
            last_err = (r.stderr or r.stdout or "")[-1500:]
            log.append("command failed, trying simpler variant: " + " ".join(cmd[1:6]) + " ...")
        except Exception as e:
            last_err = str(e)[-800:]
        if os.path.exists(out_png):
            os.remove(out_png)
    return False, last_err


ZOOM_MAX = 1.28


def render_video(job_dir, out_path, w, h, n, sec, motion, log):
    """Animate keyframes with ffmpeg zoompan + crossfade into final mp4."""
    seg_dir = os.path.join(job_dir, "segs")
    os.makedirs(seg_dir, exist_ok=True)
    fade = round(min(0.8, sec / 3.0), 2)
    frames_per_seg = int(round(sec * FPS))

    def motion_expr(i):
        if motion == "auto":
            kinds = ["zoom_in", "zoom_out", "pan_right", "pan_left"]
            kind = kinds[i % 4]
        else:
            kind = motion
        rate = (ZOOM_MAX - 1.0) / max(frames_per_seg - 1, 1)
        if kind == "zoom_in":
            return "z='min(zoom+%.5f,%.2f)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'" % (rate, ZOOM_MAX)
        if kind == "zoom_out":
            return ("z='if(lte(on,1),%.2f,max(1.001,zoom-%.5f))'"
                    ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'" % (ZOOM_MAX, rate))
        if kind == "pan_right":
            return ("z='1.25':x='(iw-iw/zoom)*on/%d':y='(ih-ih/zoom)/2'"
                    % max(frames_per_seg - 1, 1))
        return ("z='1.25':x='(iw-iw/zoom)*(1-on/%d)':y='(ih-ih/zoom)/2'"
                % max(frames_per_seg - 1, 1))

    # 1. one animated segment per keyframe
    segs = []
    for i in range(n):
        kf = os.path.join(job_dir, "kf_%03d.png" % i)
        seg = os.path.join(seg_dir, "seg_%03d.mp4" % i)
        vf = ("scale=1536:1536:force_original_aspect_ratio=increase,"
              "crop=1536:1536,"
              "zoompan=%s:d=%d:s=%dx%d:fps=%d"
              % (motion_expr(i), frames_per_seg, w, h, FPS))
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", kf, "-vf", vf,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
             "-pix_fmt", "yuv420p", seg],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ffmpeg segment failed: " + r.stderr[-600:])
        segs.append(seg)

    # 2. blend segments with crossfades
    if n == 1:
        shutil.copyfile(segs[0], out_path)
        return
    parts, prev, off = [], "[0:v]", round(sec - fade, 2)
    for i in range(1, n):
        label = "[v%d]" % i
        parts.append("%s[%d:v]xfade=transition=fade:duration=%.2f:offset=%.2f%s"
                     % (prev, i, fade, off, label))
        prev = label
        off += sec - fade
    final = os.path.join(job_dir, "final.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"] +
        sum([["-i", s] for s in segs], []) +
        ["-filter_complex", ";".join(parts), "-map", prev,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", final],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg blend failed: " + r.stderr[-600:])
    shutil.move(final, out_path)


def generation_worker(job):
    p = job["params"]
    jid = job["id"]
    job_dir = os.path.join(WORK_DIR, jid)
    os.makedirs(job_dir, exist_ok=True)
    log = job["log"]
    w = h = p["resolution"]
    try:
        if AI_MODE:
            prof = model_profile(MODEL)
            threads = max(1, (os.cpu_count() or 4) - 1)
            for i in range(p["frames"]):
                job["stage"] = "generating keyframe %d of %d (AI on device)" % (i + 1, p["frames"])
                prompt = p["prompt"]
                if p.get("evolution"):
                    prompt += ", " + SHOT_VARIATIONS[i % len(SHOT_VARIATIONS)]
                out_png = os.path.join(job_dir, "kf_%03d.png" % i)
                base_cmd = build_sd_cmd(MODEL, prompt, p.get("negative", ""), out_png,
                                        w, h, random.randint(0, 10**9), prof, threads)
                # fallback chain for CLI differences across sd.cpp versions
                simpler = [c for c in base_cmd if c not in ("--type", "q8_0")]
                minimal = [c for c in simpler if not (c.startswith("--taesd"))]
                minimal = [c for c in minimal if c != TAESD and c != "--sampling-method" and c not in prof_samplers()]
                ok, err = run_sd([base_cmd, simpler, minimal], out_png, log)
                if not ok:
                    raise RuntimeError("image generation failed:\n" + err[-800:])
                job["frames_done"] = i + 1
                job["progress"] = 80.0 * (i + 1) / p["frames"]
        else:
            for i in range(p["frames"]):
                job["stage"] = "demo frame %d of %d (install a model for real AI images)" % (i + 1, p["frames"])
                make_demo_frame(os.path.join(job_dir, "kf_%03d.png" % i),
                                w, h, i, p["frames"], p["prompt"])
                time.sleep(0.3)
                job["frames_done"] = i + 1
                job["progress"] = 80.0 * (i + 1) / p["frames"]

        job["stage"] = "rendering video with ffmpeg"
        out_path = os.path.join(OUTPUTS_DIR, jid + ".mp4")
        render_video(job_dir, out_path, w, h, p["frames"], p["sec"],
                     p.get("motion", "auto"), log)
        job["progress"] = 100.0
        job["status"] = "done"
        job["stage"] = "ready"
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)[-1200:]
        log.append(str(e)[-400:])


def prof_samplers():
    return ["euler", "euler a", "lcm"]


# ---------------------------------------------------------------- flask app

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_file(os.path.join(APP_DIR, "public", "index.html"))


@app.route("/webgpu/<path:sub>")
def webgpu_static(sub):
    """Serve WebGPU engine assets (ort vendor + onnx models) locally - no CDN."""
    p = os.path.normpath(os.path.join(WEBGPU_DIR, sub))
    if not p.startswith(os.path.abspath(WEBGPU_DIR)) or not os.path.isfile(p):
        abort(404)
    mime = {
        ".js": "application/javascript", ".mjs": "application/javascript",
        ".wasm": "application/wasm", ".json": "application/json",
        ".txt": "text/plain", ".onnx": "application/octet-stream",
    }.get(os.path.splitext(p)[1], "application/octet-stream")
    return send_file(p, mimetype=mime, conditional=True)


@app.route("/webgpu-engine.js")
def webgpu_engine_js():
    return send_file(os.path.join(APP_DIR, "public", "webgpu-engine.js"),
                     mimetype="application/javascript")


@app.route("/api/status")
def api_status():
    return jsonify({
        "ai_mode": AI_MODE,
        "sd_bin": bool(SD_BIN),
        "model": os.path.basename(MODEL) if MODEL else None,
        "taesd": bool(TAESD),
        "webgpu_files": WEBGPU_FILES,
    })


def webgpu_waiter_worker(job):
    """Waits for the browser to upload GPU-generated keyframes, then renders."""
    p = job["params"]
    jid = job["id"]
    job_dir = os.path.join(WORK_DIR, jid)
    os.makedirs(job_dir, exist_ok=True)
    try:
        deadline = time.time() + 4 * 3600
        while job["frames_done"] < p["frames"] and time.time() < deadline:
            if job.get("aborted"):
                raise RuntimeError("generation was aborted")
            time.sleep(1.5)
        if job["frames_done"] < p["frames"]:
            raise RuntimeError("timed out waiting for GPU keyframes from the browser")
        job["stage"] = "rendering video with ffmpeg"
        out_path = os.path.join(OUTPUTS_DIR, jid + ".mp4")
        render_video(job_dir, out_path, p["resolution"], p["resolution"],
                     p["frames"], p["sec"], p.get("motion", "auto"), job["log"])
        job["progress"] = 100.0
        job["status"] = "done"
        job["stage"] = "ready"
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)[-1200:]


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "please type a prompt first"}), 400
    if not BUSY_LOCK.acquire(blocking=False):
        return jsonify({"error": "a generation is already running - wait for it to finish"}), 409

    params = {
        "prompt": prompt[:500],
        "negative": (data.get("negative") or "blurry, low quality, watermark, text")[:300],
        "frames": min(max(int(data.get("frames", 4)), 1), 10),
        "sec": min(max(float(data.get("sec", 3.5)), 1.5), 8.0),
        "resolution": int(data.get("resolution", 384)),
        "motion": data.get("motion", "auto"),
        "evolution": bool(data.get("evolution", True)),
    }
    if params["resolution"] not in (320, 384, 448, 512):
        params["resolution"] = 384
    engine = data.get("engine", "auto")
    # effective engine: webgpu needs both the assets and a GPU-capable browser
    if engine == "webgpu" and not WEBGPU_FILES:
        engine = "cpu" if AI_MODE else "demo"
    if engine == "auto":
        engine = "webgpu" if WEBGPU_FILES else ("cpu" if AI_MODE else "demo")
    if engine == "cpu" and not AI_MODE:
        engine = "demo"
    params["engine"] = engine
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "status": "running", "stage": "starting",
           "progress": 0.0, "frames_done": 0, "total_frames": params["frames"],
           "t0": time.time(), "error": None, "params": params, "log": [],
           "aborted": False}
    with JOBS_LOCK:
        JOBS[jid] = job
    threading.Thread(target=guarded_worker, args=(job,), daemon=True).start()
    return jsonify({"job_id": jid, "engine": engine})


def guarded_worker(job):
    try:
        if job["params"].get("engine") == "webgpu":
            webgpu_waiter_worker(job)
        else:
            generation_worker(job)
    finally:
        BUSY_LOCK.release()


@app.route("/api/upload_frame/<jid>", methods=["POST"])
def api_upload_frame(jid):
    """Receive one GPU-generated keyframe PNG from the browser."""
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if not job or job["params"].get("engine") != "webgpu":
        abort(404)
    try:
        idx = int(request.headers.get("X-Frame-Index", "-1"))
    except ValueError:
        abort(400)
    if not (0 <= idx < job["total_frames"]):
        abort(400)
    job_dir = os.path.join(WORK_DIR, jid)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "kf_%03d.png" % idx), "wb") as f:
        f.write(request.get_data())
    with JOBS_LOCK:
        job["frames_done"] = max(job["frames_done"], idx + 1)
        job["progress"] = 80.0 * job["frames_done"] / job["total_frames"]
        job["stage"] = ("GPU keyframe %d of %d done" %
                        (job["frames_done"], job["total_frames"]))
    return jsonify({"ok": True, "frames_done": job["frames_done"]})


@app.route("/api/abort/<jid>", methods=["POST"])
def api_abort(jid):
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if not job:
        abort(404)
    job["aborted"] = True
    return jsonify({"ok": True})


@app.route("/api/job/<jid>")
def api_job(jid):
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if not job:
        abort(404)
    return jsonify(job_snapshot(job))


@app.route("/api/frame/<jid>/<int:i>.png")
def api_frame(jid, i):
    job_dir = os.path.join(WORK_DIR, jid)
    # finished jobs are cleaned up - no frames to show
    p = os.path.join(job_dir, "kf_%03d.png" % i)
    if not os.path.isfile(p):
        abort(404)
    return send_file(p, mimetype="image/png")


@app.route("/api/video/<name>")
def api_video(name):
    if not re.fullmatch(r"[0-9a-f]{12}\.mp4", name):
        abort(404)
    p = os.path.join(OUTPUTS_DIR, name)
    if not os.path.isfile(p):
        abort(404)
    return send_file(p, mimetype="video/mp4", conditional=True)


@app.route("/api/gallery")
def api_gallery():
    vids = []
    for f in sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*.mp4")),
                    key=os.path.getmtime, reverse=True)[:30]:
        vids.append({"name": os.path.basename(f),
                     "size_mb": round(os.path.getsize(f) / 1e6, 1),
                     "created": time.strftime("%d/%m %H:%M", time.localtime(os.path.getmtime(f)))})
    return jsonify(vids)


if __name__ == "__main__":
    print("=" * 52)
    print("  LocalAI Video Studio")
    print("  mode: %s" % ("AI (%s)" % os.path.basename(MODEL) if AI_MODE else
                          "DEMO (no model installed yet)"))
    print("  open in your browser:  http://localhost:%d" % PORT)
    print("=" * 52)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
