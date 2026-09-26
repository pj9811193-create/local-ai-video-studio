/**
 * WebGPU engine for LocalAI Video Studio.
 *
 * Runs SD-Turbo entirely in the browser via onnxruntime-web's WebGPU
 * execution provider, using the phone/desktop GPU through Chrome.
 * Ported from Microsoft's onnxruntime-inference-examples (js/sd-turbo),
 * MIT licensed. No internet needed: all model files are served by the
 * local Flask server (downloaded once by setup.sh).
 *
 * Exposes:
 *   const engine = new SDTurboWebGPU()
 *   await engine.init(onProgress)     // loads ort + models (~2.4 GB)
 *   const canvas = await engine.generate(prompt, onStep)
 */

"use strict";

/* ------------------------------------------------------------------ *
 * CLIP tokenizer (byte-level BPE, </w> scheme) - standard CLIP scheme.
 * Loads vocab.json + merges.txt served from the local server.
 * Verified token-identical to HuggingFace CLIPTokenizer.
 * ------------------------------------------------------------------ */
class ClipTokenizer {
  constructor() {
    this.vocab = null;        // token string -> id
    this.merges = null;       // pair -> rank
  }

  async load(baseUrl) {
    const [vocabJson, mergesText] = await Promise.all([
      fetch(baseUrl + "vocab.json").then(r => {
        if (!r.ok) throw new Error("vocab.json " + r.status);
        return r.json();
      }),
      fetch(baseUrl + "merges.txt").then(r => {
        if (!r.ok) throw new Error("merges.txt " + r.status);
        return r.text();
      }),
    ]);
    this.vocab = vocabJson;
    this.merges = new Map();
    mergesText.split("\n").forEach((line, i) => {
      const parts = line.split(" ");
      if (parts.length === 2) this.merges.set(parts[0] + "," + parts[1], i);
    });
  }

  // GPT-2 byte-to-unicode map (deterministic, no download needed)
  static _bytesToUnicode() {
    const bs = [];
    for (let i = 0; i < 256; i++) bs.push(i);
    const cs = bs.map(b => (b >= 33 && b <= 126) || (b >= 161 && b <= 172)
      || (b >= 174 && b <= 255) ? b : null);
    let n = 0;
    for (let i = 0; i < 256; i++) {
      if (cs[i] === null) { cs[i] = 256 + n; n++; }
    }
    const map = {};
    bs.forEach((b, i) => { map[b] = String.fromCharCode(cs[i]); });
    return map;
  }

  _bpe(token) {
    // CLIP scheme: the last character of each word carries the "</w>" marker
    if (this._cache.has(token)) return this._cache.get(token);
    const word = token.split("");
    word[word.length - 1] += "</w>";
    for (;;) {
      let bestRank = Infinity, bestPair = null;
      for (const pair of this._getPairs(word)) {
        const rank = this.merges.get(pair) ?? Infinity;
        if (rank < bestRank) { bestRank = rank; bestPair = pair; }
      }
      if (bestPair === null || bestRank === Infinity) break;
      this._mergeWord(word, bestPair);
    }
    this._cache.set(token, word);
    return word;
  }

  _getPairs(word) {
    const pairs = new Set();
    for (let i = 0; i < word.length - 1; i++) {
      pairs.add(word[i] + "," + word[i + 1]);
    }
    return pairs;
  }

  _mergeWord(word, pairStr) {
    const [first, second] = pairStr.split(",");
    const out = [];
    let i = 0;
    while (i < word.length) {
      if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
        out.push(first + second);
        i += 2;
      } else {
        out.push(word[i]);
        i += 1;
      }
    }
    word.length = 0;
    word.push(...out);
  }

  /** Encode to CLIP token ids, fixed length 77 (BOS ... EOS pad 0). */
  encode(text, maxLength = 77) {
    if (!this._cache) this._cache = new Map();
    const BOS = 49406, EOS = 49407, PAD = 0;
    // CLIP: strip, collapse whitespace, lowercase
    const clean = String(text).replace(/\s+/g, " ").trim().toLowerCase();
    const re = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/gu;
    const b2u = ClipTokenizer._bytesToUnicode();
    const tokens = [];
    for (const m of clean.match(re) || []) {
      const bytes = new TextEncoder().encode(m);
      let piece = "";
      for (const b of bytes) piece += b2u[b];
      tokens.push(piece);
    }
    let ids = [];
    for (const t of tokens) {
      for (const piece of this._bpe(t)) ids.push(this.vocab[piece]);
    }
    // BOS/EOS + pad/truncate to maxLength (pad token 0, like the MS example)
    ids = [BOS, ...ids.slice(0, maxLength - 2), EOS];
    const out = new Int32Array(maxLength);
    out.set(ids.slice(0, maxLength));
    for (let i = ids.length; i < maxLength; i++) out[i] = PAD;
    return out;
  }
}

/* ------------------------------------------------------------------ *
 * SD-Turbo pipeline on WebGPU via onnxruntime-web.
 * ------------------------------------------------------------------ */
class SDTurboWebGPU {
  constructor() {
    this.ready = false;
    this.sigma = 14.6146;
    this.gamma = 0;
    this.vaeScalingFactor = 0.18215;
  }

  get supported() {
    return typeof navigator !== "undefined" && !!navigator.gpu;
  }

  async init(onProgress) {
    if (this.ready) return;
    const say = m => { if (onProgress) onProgress(m); };
    if (!this.supported) throw new Error("WebGPU is not supported by this browser");
    if (typeof ort === "undefined") throw new Error("onnxruntime-web not loaded (re-run setup)");
    ort.env.wasm.wasmPaths = "/webgpu/vendor/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    const base = "/webgpu/models/sd-turbo/";
    const mkOpt = (freeDims) => ({
      executionProviders: ["webgpu"],
      enableMemPattern: false,
      enableCpuMemArena: false,
      freeDimensionOverrides: freeDims,
      extra: {
        session: {
          disable_prepacking: "1",
          use_device_allocator_for_initializers: "1",
          use_ort_model_bytes_directly: "1",
          use_ort_model_bytes_for_initializers: "1",
        },
      },
    });

    say("loading text encoder…");
    this.textEncoder = await ort.InferenceSession.create(
      base + "text_encoder/model.onnx",
      mkOpt({ batch_size: 1 }));

    say("loading unet…");
    this.unet = await ort.InferenceSession.create(
      base + "unet/model.onnx",
      mkOpt({ batch_size: 1, num_channels: 4, height: 64, width: 64, sequence_length: 77 }));

    say("loading image decoder…");
    this.vae = await ort.InferenceSession.create(
      base + "vae_decoder/model.onnx",
      mkOpt({ batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 }));

    say("loading tokenizer…");
    this.tokenizer = new ClipTokenizer();
    await this.tokenizer.load("/webgpu/models/clip-tokenizer/");

    this.ready = true;
    say("GPU engine ready");
  }

  _randnLatents(shape, sigma) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const u = Math.random(), v = Math.random();
      data[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * sigma;
    }
    return new ort.Tensor(data, shape);
  }

  _scaleModelInputs(t) {
    const dIn = t.data, dOut = new Float32Array(dIn.length);
    const divi = Math.sqrt(this.sigma ** 2 + 1);
    for (let i = 0; i < dIn.length; i++) dOut[i] = dIn[i] / divi;
    return new ort.Tensor(dOut, t.dims);
  }

  _step(modelOutput, sample) {
    const dOut = new Float32Array(modelOutput.data.length);
    const sigmaHat = this.sigma * (this.gamma + 1);
    const s = sample.data, m = modelOutput.data;
    for (let i = 0; i < dOut.length; i++) {
      const predOriginal = s[i] - sigmaHat * m[i];
      const derivative = (s[i] - predOriginal) / sigmaHat;
      const dt = 0 - sigmaHat;
      dOut[i] = (s[i] + derivative * dt) / this.vaeScalingFactor;
    }
    return new ort.Tensor(dOut, modelOutput.dims);
  }

  /**
   * Generate one 512x512 image. Returns a canvas with the result.
   * promptVariation is appended to the prompt (scene variety).
   */
  async generate(prompt, onStep) {
    if (!this.ready) throw new Error("engine not initialized");
    const say = m => { if (onStep) onStep(m); };

    const ids = this.tokenizer.encode(prompt);
    const idTensor = new ort.Tensor("int32", ids, [1, ids.length]);

    say("text encoder");
    const teIn = this.textEncoder.inputNames[0];
    const teOut = this.textEncoder.outputNames[0];
    const teResults = await this.textEncoder.run({ [teIn]: idTensor });
    const lastHiddenState = teResults[teOut];

    say("unet");
    const latentShape = [1, 4,64,64];
    const latent = this._randnLatents(latentShape, this.sigma);
    const uIn = this.unet.inputNames;   // [sample, timestep, encoder_hidden_states]
    const uOut = this.unet.outputNames[0];
    const uResults = await this.unet.run({
      [uIn[0]]: this._scaleModelInputs(latent),
      [uIn[1]]: new ort.Tensor("int64", [999n], [1]),
      [uIn[2]]: lastHiddenState,
    });
    const newLatents = this._step(uResults[uOut], latent);

    say("image decoder");
    const vIn = this.vae.inputNames[0];
    const vOut = this.vae.outputNames[0];
    const vResults = await this.vae.run({ [vIn]: newLatents });
    const imgTensor = vResults[vOut];

    // draw to canvas
    const pix = imgTensor.data;
    for (let i = 0; i < pix.length; i++) {
      let x = pix[i] / 2 + 0.5;
      pix[i] = x < 0 ? 0 : (x > 1 ? 1 : x);
    }
    const imageData = imgTensor.toImageData({ tensorLayout: "NCWH", format: "RGB" });
    const canvas = document.createElement("canvas");
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext("2d").putImageData(imageData, 0, 0);
    return canvas;
  }
}

window.SDTurboWebGPU = SDTurboWebGPU;


/* ================================================================
 *  Our OWN tiny AI model (~0.9M params, trained from scratch by
 *  scripts/train_custom_model.py - no third-party weights). It paints
 *  simple stylized scene images at 64x64, one of 12 categories
 *  picked from the prompt keywords. Runs on any browser via wasm.
 * ================================================================ */
class LocalTinyModel {
  constructor() { this.ready = false; this.sess = null; }

  async init(onMsg) {
    onMsg('loading our own tiny model (~4 MB)...');
    const [buf, labels] = await Promise.all([
      fetch('/webgpu/models/custom/model.onnx').then(r => {
        if (!r.ok) throw new Error('custom model not installed (re-run setup)');
        return r.arrayBuffer();
      }),
      fetch('/webgpu/models/custom/labels.json').then(r => r.json()),
    ]);
    this.labels = labels;
    this.sess = await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    // noise schedule (must mirror training: 500 steps, linear 1e-4..0.02)
    this.abar = [];
    let a = 1;
    for (let i = 0; i < 500; i++) {
      a *= (1 - (1e-4 + (0.02 - 1e-4) * i / 499));
      this.abar.push(a);
    }
    this.ready = true;
    onMsg('own model ready');
  }

  pickClass(prompt) {
    const p = prompt.toLowerCase();
    let best = -1, bestHits = 0;
    for (const [cls, words] of Object.entries(this.labels.keywords || {})) {
      let hits = 0;
      for (const w of words) if (p.includes(w)) hits += w.length;
      if (hits > bestHits) { bestHits = hits; best = this.labels.classes.indexOf(cls); }
    }
    if (best < 0) best = this.labels.classes.indexOf('abstract');
    return best < 0 ? 0 : best;
  }

  async generate(prompt, seed, onStep) {
    const S = 64, steps = 28, cls = this.pickClass(prompt);
    // seeded gaussian noise (Box-Muller + mulberry32)
    const rng = (s => () => {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    })(seed);
    const x = new Float32Array(3 * S * S);
    for (let i = 0; i < x.length; i += 2) {
      const u = Math.max(1e-9, rng()), v = rng(), r = Math.sqrt(-2 * Math.log(u));
      x[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < x.length) x[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    // DDIM denoising
    for (let s = 0; s < steps; s++) {
      const t = Math.round(499 * (1 - s / (steps - 1)));
      const out = await this.sess.run({
        x: new ort.Tensor('float32', x, [1, 3, S, S]),
        t: new ort.Tensor('int64', BigInt64Array.from([BigInt(t)]), [1]),
        class: new ort.Tensor('int64', BigInt64Array.from([BigInt(cls)]), [1]),
      });
      const eps = out.eps.data;
      const ab = this.abar[t];
      const tPrev = s + 1 < steps ? Math.round(499 * (1 - (s + 1) / (steps - 1))) : -1;
      const abp = tPrev < 0 ? 1.0 : this.abar[tPrev];
      const sa = Math.sqrt(1 - ab), sb = Math.sqrt(ab), spa = Math.sqrt(1 - abp), sp = Math.sqrt(abp);
      for (let i = 0; i < x.length; i++) {
        let x0 = (x[i] - sa * eps[i]) / sb;
        x0 = Math.max(-1.2, Math.min(1.2, x0));
        x[i] = sp * x0 + spa * eps[i];
      }
      if (onStep && (s % 7 === 0 || s === steps - 1)) onStep(`denoising ${s + 1}/${steps}`);
    }
    // draw at 64x64, upscale to 512 with smoothing
    const small = document.createElement('canvas');
    small.width = S; small.height = S;
    const sctx = small.getContext('2d');
    const img = sctx.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      for (let c = 0; c < 3; c++) {
        img.data[i * 4 + c] = Math.max(0, Math.min(255, (x[c * S * S + i] + 1) * 127.5));
      }
      img.data[i * 4 + 3] = 255;
    }
    sctx.putImageData(img, 0, 0);
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(small, 0, 0, 512, 512);
    return canvas;
  }
}
