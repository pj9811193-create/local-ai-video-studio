#!/usr/bin/env python3
"""
train_custom_model.py - build LocalAI's OWN tiny image model (~4 MB).

Trains a small conditional diffusion model from scratch (our architecture,
our procedurally generated training set - no one else's weights), then
exports it to ONNX for the browser (onnxruntime-web) and writes a sample
grid so quality can be checked by eye.

It is NOT a replacement for SD-Turbo: it paints simple stylized scene
images at 64x64 (upscaled by the app), one of 12 categories picked from
the prompt. That is the honest maximum for a model this size trained on
a CPU.

Usage:
    python3 train_custom_model.py [--steps 3000] [--n-per-class 500]
                                  [--out out] [--batch 32]
Outputs (in --out):
    model.onnx    the model (float32)
    labels.json   categories + keyword hints for the app
    samples.png   one generated image per category (via the exported ONNX,
                  mirroring the browser engine's DDIM exactly)
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 64
CLASSES = ["sunset", "ocean", "forest", "mountains", "night", "city",
           "desert", "snow", "meadow", "space", "fire", "abstract"]
KEYWORDS = {
    "sunset": ["sunset", "dusk", "sunrise", "dawn", "evening", "golden"],
    "ocean": ["ocean", "sea", "beach", "water", "wave", "island", "shore",
              "waterfall", "river", "lake"],
    "forest": ["forest", "tree", "jungle", "wood", "bamboo"],
    "mountains": ["mountain", "hill", "peak", "valley", "canyon", "cliff"],
    "night": ["night", "star", "moon", "midnight", "dark sky"],
    "city": ["city", "building", "street", "urban", "skyline", "town",
             "neon", "bridge"],
    "desert": ["desert", "sand", "dune", "sahara", "cactus", "arid"],
    "snow": ["snow", "winter", "ice", "frost", "arctic", "cold"],
    "meadow": ["meadow", "flower", "field", "spring", "garden", "grass",
               "bloom"],
    "space": ["space", "galaxy", "nebula", "cosmos", "planet", "universe",
              "star system", "astro"],
    "fire": ["fire", "flame", "lava", "volcano", "ember", "burning"],
    "abstract": [],
}

# ------------------------------------------------------------------ dataset


def _smooth_noise(rng, w, h, octaves=4):
    """Smooth perlin-ish noise field in [0,1]."""
    total = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    for o in range(octaves):
        gw, gh = max(2, w >> o), max(2, h >> o)
        small = rng.random((gh, gw), dtype=np.float32)
        big = np.array(np.kron(small, np.ones((max(1, h // gh),
                                                max(1, w // gw))))[:h, :w])
        total += amp * big / (big.max() + 1e-9)
        amp *= 0.5
    lo, hi = total.min(), total.max()
    return (total - lo) / (hi - lo + 1e-9)


def _vgrad(rng, top, bottom):
    """Vertical gradient [h,3] -> [h,w,3] via numpy broadcast later."""
    t = np.linspace(0, 1, IMG)[:, None, None]
    return top[None, None, :] * (1 - t) + bottom[None, None, :] * t


def _finish(img):
    return np.clip(img, 0, 1)


def gen_sunset(rng):
    top = np.array([0.25, 0.15, 0.45]) + rng.normal(0, 0.05, 3)
    bot = np.array([1.0, 0.45, 0.15]) + rng.normal(0, 0.05, 3)
    img = np.repeat(_vgrad(rng, top, bot), IMG, axis=1)
    x, y = rng.integers(10, IMG - 10), rng.integers(30, 50)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    sun = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 8.0 ** 2)))
    img += sun[..., None] * np.array([1.0, 0.85, 0.5])
    img[int(y) + 4:, :] *= 0.25          # dark ground silhouette
    return _finish(img)


def gen_ocean(rng):
    top = np.array([0.45, 0.65, 0.9]) + rng.normal(0, 0.05, 3)
    bot = np.array([0.05, 0.2, 0.45]) + rng.normal(0, 0.04, 3)
    img = np.repeat(_vgrad(rng, top, bot), IMG, axis=1)
    hz = rng.integers(24, 36)
    waves = np.tile(np.sin(np.linspace(0, 40, IMG - hz) * rng.uniform(1, 2)
                           + rng.uniform(0, 6))[:, None], (1, IMG))
    img[hz:, :] *= (0.75 + 0.25 * np.abs(waves))[..., None]
    x = rng.integers(5, IMG - 5)
    glint = np.exp(-((np.arange(IMG) - x) ** 2) / 40.0) * 0.8
    img[hz:, :] += glint[None, :, None] * 0.4
    return _finish(img)


def gen_forest(rng):
    img = np.repeat(_vgrad(rng, np.array([0.1, 0.25, 0.12]),
                           np.array([0.02, 0.08, 0.03])), IMG, axis=1)
    n = rng.integers(14, 24)
    for _ in range(n):
        x = rng.integers(0, IMG)
        w = rng.integers(1, 3)
        img[:, x:x + w] *= 0.35 + 0.3 * rng.random()   # trunks
    canopy = _smooth_noise(rng, IMG, IMG, 5)[..., None]
    img += canopy * np.array([0.1, 0.25, 0.05])
    return _finish(img)


def gen_mountains(rng):
    img = np.repeat(_vgrad(rng, np.array([0.5, 0.65, 0.9]),
                           np.array([0.8, 0.85, 0.95])), IMG, axis=1)
    color = np.array([0.15, 0.25, 0.35])
    for layer, shade in ((0.7, 0.9), (0.5, 0.65), (0.3, 0.4)):
        base = np.zeros(IMG)
        n = rng.integers(3, 6)
        xs = np.sort(rng.integers(0, IMG, n))
        for i in range(n - 1):
            seg = slice(xs[i], xs[i + 1] if i < n - 2 else IMG)
            peak = rng.integers(14, 42)
            base[seg] = np.linspace(peak, 0, seg.stop - seg.start)
        ridge = np.maximum.accumulate(np.maximum(base, 0)) * layer
        mask = np.arange(IMG)[None, :] < ridge[:, None]
        img[mask] = color * shade
    return _finish(img)


def gen_night(rng):
    img = np.repeat(_vgrad(rng, np.array([0.02, 0.03, 0.12]),
                           np.array([0.0, 0.01, 0.05])), IMG, axis=1)
    n = rng.integers(60, 140)
    ys, xs = rng.integers(0, IMG, n), rng.integers(0, IMG, n)
    b = rng.random(n) * 0.9 + 0.1
    img[ys, xs] += b[:, None]
    mx, my = rng.integers(8, IMG - 8), rng.integers(6, 22)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img += np.exp(-(((xx - mx) ** 2 + (yy - my) ** 2) / 18.0))[..., None] * 0.9
    return _finish(img)


def gen_city(rng):
    img = np.repeat(_vgrad(rng, np.array([0.05, 0.05, 0.18]),
                           np.array([0.1, 0.05, 0.1])), IMG, axis=1)
    x = 0
    while x < IMG:
        w = rng.integers(5, 14)
        hh = rng.integers(18, 55)
        img[IMG - hh:IMG, x:x + w] = 0.08          # building silhouette
        for wy in range(IMG - hh + 3, IMG - 4, 5):
            for wx in range(x + 2, x + w - 2, 4):
                if rng.random() < 0.55:
                    c = rng.uniform(0.6, 1.0)
                    img[wy:wy + 2, wx:wx + 2] = [c, c * 0.85, c * 0.5]
        x += w + rng.integers(1, 3)
    return _finish(img)


def gen_desert(rng):
    img = np.repeat(_vgrad(rng, np.array([0.95, 0.75, 0.45]),
                           np.array([0.75, 0.5, 0.25])), IMG, axis=1)
    for shade, off in ((1.05, 0.55), (0.9, 0.7)):
        base = np.sin(np.linspace(0, 6.28, IMG) + off)
        yy = (base * 6 + 40).astype(int)
        mask = np.arange(IMG)[None, :] >= yy[:, None] - 8
        img[mask] *= shade
    return _finish(img)


def gen_snow(rng):
    img = np.repeat(_vgrad(rng, np.array([0.7, 0.78, 0.9]),
                           np.array([0.92, 0.95, 1.0])), IMG, axis=1)
    drift = _smooth_noise(rng, IMG, IMG, 3)[..., None]
    img = img * 0.85 + drift * 0.15
    n = rng.integers(80, 200)
    ys, xs = rng.integers(0, IMG, n), rng.integers(0, IMG, n)
    img[ys, xs] = 1.0
    return _finish(img)


def gen_meadow(rng):
    img = np.repeat(_vgrad(rng, np.array([0.55, 0.75, 0.95]),
                           np.array([0.2, 0.5, 0.15])), IMG, axis=1)
    n = rng.integers(30, 70)
    ys = rng.integers(36, IMG, n)
    xs = rng.integers(0, IMG, n)
    palette = np.array([[0.95, 0.3, 0.35], [0.95, 0.8, 0.25],
                        [0.85, 0.5, 0.9], [0.95, 0.95, 0.9], [0.5, 0.4, 0.9]])
    cols = palette[rng.integers(0, len(palette), n)]
    for y, x, c in zip(ys, xs, cols):
        r = rng.integers(1, 3)
        img[max(0, y - r):y + r, max(0, x - r):x + r] = c
    return _finish(img)


def gen_space(rng):
    img = np.zeros((IMG, IMG, 3), dtype=np.float32)
    for _ in range(rng.integers(2, 4)):
        cx, cy = rng.integers(0, IMG, 2)
        col = rng.random(3) * 0.9
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        img += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) /
                        (2 * rng.uniform(8, 20) ** 2)))[..., None] * col * 0.8
    n = rng.integers(80, 160)
    ys, xs = rng.integers(0, IMG, n), rng.integers(0, IMG, n)
    img[ys, xs] += (rng.random(n) * 0.9 + 0.1)[:, None]
    return _finish(img)


def gen_fire(rng):
    img = np.zeros((IMG, IMG, 3), dtype=np.float32)
    noise = _smooth_noise(rng, IMG, IMG, 5)
    for y in range(IMG):
        rise = (y / IMG) ** 1.5
        img[y, :, 0] = np.clip(noise[y] * 1.3 * rise + 0.05, 0, 1) * 0.9
        img[y, :, 1] = np.clip(noise[y] * 0.9 * rise - 0.15, 0, 1) * 0.45
        img[y, :, 2] = np.clip(noise[y] * 0.8 * rise - 0.3, 0, 1) * 0.15
    return _finish(img)


def gen_abstract(rng):
    img = np.zeros((IMG, IMG, 3), dtype=np.float32)
    for _ in range(rng.integers(3, 6)):
        cx, cy = rng.integers(0, IMG, 2)
        col = rng.random(3)
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        img += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) /
                        (2 * rng.uniform(10, 30) ** 2)))[..., None] * col
    return _finish(img)


GENS = [gen_sunset, gen_ocean, gen_forest, gen_mountains, gen_night, gen_city,
        gen_desert, gen_snow, gen_meadow, gen_space, gen_fire, gen_abstract]


def make_dataset(n_per_class, seed=7):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for ci, gen in enumerate(GENS):
        for _ in range(n_per_class):
            xs.append(gen(rng).transpose(2, 0, 1))   # [3,H,W]
            ys.append(ci)
    # store as uint8 (5x less RAM on small machines); float on demand
    x = torch.tensor((np.stack(xs) * 255), dtype=torch.uint8)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


# ------------------------------------------------------------------ model

class SinEmb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim=128):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = (nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity())

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """~0.9M-param conditional UNet for 64x64 images."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.time = nn.Sequential(SinEmb(128), nn.Linear(128, 128), nn.SiLU(),
                                  nn.Linear(128, 128))
        self.cls = nn.Embedding(n_classes, 128)
        self.inp = nn.Conv2d(3, 32, 3, padding=1)
        self.e0 = ResBlock(32, 32)
        self.e1 = ResBlock(32, 64)
        self.e2 = ResBlock(64, 128)
        self.mid = ResBlock(128, 128)
        self.d1 = ResBlock(128 + 64, 64)
        self.d0 = ResBlock(64 + 32, 32)
        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x, t, c):
        e = self.time(t) + self.cls(c)
        h0 = self.e0(self.inp(x), e)
        h1 = self.e1(F.avg_pool2d(h0, 2), e)
        h2 = self.e2(F.avg_pool2d(h1, 2), e)
        m = self.mid(h2, e)
        u = F.interpolate(m, scale_factor=2, mode="nearest")
        u = self.d1(torch.cat([u, h1], 1), e)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.d0(torch.cat([u, h0], 1), e)
        return self.out(u)


# ------------------------------------------------------------------ train

T_MAX = 500
BETAS = torch.linspace(1e-4, 0.02, T_MAX)
ALPHAS = 1.0 - BETAS
ABAR = torch.cumprod(ALPHAS, 0)          # [T]


def sample_onnx(onnx_path, cls, seed=0, steps=28):
    """DDIM sampling via the exported ONNX - mirrors the browser engine
    (webgpu-engine.js LocalTinyModel) exactly, so the sample grid shows what
    the app will actually produce."""
    import onnxruntime as ort
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 3, IMG, IMG)).astype(np.float32)
    abar = np.cumprod(1.0 - np.linspace(1e-4, 0.02, T_MAX))
    ts = np.round(np.linspace(T_MAX - 1, 0, steps)).astype(np.int64)
    for i, t in enumerate(ts):
        eps = s.run(None, {"x": x, "t": np.array([t], np.int64),
                           "class": np.array([cls], np.int64)})[0][0]
        ab = abar[t]
        t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
        abp = 1.0 if t_prev < 0 else abar[t_prev]
        x0 = np.clip((x - np.sqrt(1 - ab) * eps) / np.sqrt(ab), -1.2, 1.2)
        x = (np.sqrt(abp) * x0 + np.sqrt(1 - abp) * eps).astype(np.float32)
    return torch.tensor(x[0])


def export_onnx(model, path):
    model = model.eval()
    x = torch.zeros(1, 3, IMG, IMG)
    t = torch.zeros(1, dtype=torch.int64)
    c = torch.zeros(1, dtype=torch.int64)
    torch.onnx.export(model, (x, t, c), path,
                      input_names=["x", "t", "class"],
                      output_names=["eps"],
                      dynamic_axes={"x": {0: "batch"}, "eps": {0: "batch"}},
                      opset_version=17)


def grid_png(xs, path):
    from PIL import Image
    n = len(xs)
    imgs = []
    for x in xs:
        arr = ((x.clamp(-1, 1) + 1) / 2).numpy()
        arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((128, 128), Image.BILINEAR)
        imgs.append(np.asarray(im))
    rows = int(math.ceil(n / 4))
    sheet = np.zeros((rows * 128, 4 * 128, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, col = divmod(i, 4)
        sheet[r * 128:(r + 1) * 128, col * 128:(col + 1) * 128] = im
    Image.fromarray(sheet).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 2))

    print("[1/4] generating training data (our own procedural set)...")
    x, y = make_dataset(args.n_per_class, seed=args.seed)
    print("      %d images, %d classes" % (len(x), len(CLASSES)))

    print("[2/4] training %d steps..." % args.steps)
    model = TinyUNet()
    print("      params: %.2f M" % (sum(p.numel() for p in model.parameters())
                                     / 1e6))
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(x), args.batch)
        xb = x[idx].float() / 127.5 - 1.0     # [-1, 1]
        yb = y[idx]
        tb = torch.tensor(rng.integers(0, T_MAX, args.batch),
                          dtype=torch.int64)
        noise = torch.randn(xb.shape)
        ab = ABAR[tb][:, None, None, None]
        xt = ab.sqrt() * xb + (1 - ab).sqrt() * noise
        pred = model(xt, tb, yb)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        del xb, noise, xt, pred
        if step % max(1, args.steps // 20) == 0 or step == 1:
            print("      step %5d  loss %.4f" % (step, loss.item()), flush=True)

    print("[3/4] exporting ONNX + sample grid...")
    onnx_path = os.path.join(args.out, "model.onnx")
    export_onnx(model, onnx_path)
    grid_png([sample_onnx(onnx_path, ci, seed=100 + ci)
              for ci in range(len(CLASSES))],
             os.path.join(args.out, "samples.png"))

    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump({"classes": CLASSES, "keywords": KEYWORDS}, f, indent=1)

    mb = os.path.getsize(onnx_path) / 1048576
    print("[4/4] done. model.onnx = %.1f MB" % mb)
    print("      sanity: import with onnxruntime to verify...")
    import onnxruntime as ort
    s = ort.InferenceSession(onnx_path)
    out = s.run(None, {"x": np.zeros((1, 3, IMG, IMG), np.float32),
                       "t": np.zeros(1, np.int64),
                       "class": np.zeros(1, np.int64)})[0]
    print("      onnx output shape:", out.shape)


if __name__ == "__main__":
    main()
