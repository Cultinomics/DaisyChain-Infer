// Forward-only transformer, split at layer boundaries, for real HF models.
//
// Every matrix product goes through the verified INT8 units — the same
// block-scaled quantize -> exact LUT/DP4A multiply -> exact int32 accumulate ->
// pinned f32 epilogue as the rest of DaisyChain. Nothing here computes in
// plain float except the elementwise parts that have no matmul in them
// (norms, softmax, RoPE, the activation), exactly as the trainer does it.
//
// The pass is cut into three callable pieces so they can run on different
// machines:
//
//   embed(ids)        head stage: token (+ learned position) embedding
//   runLayers(x)      any stage:  its own contiguous layers
//   readout(x)        head stage: final norm + lm_head -> logits
//
// Architecture differences live in the spec (see arch.js), never in branches
// scattered through the maths: `spec.norm`, `spec.gated`, `spec.rope`,
// `spec.kvHeads`. A Llama block and a GPT-2 block take the same path here.
(function (root) {
  "use strict";

  let TC, V, SH;

  // ---- norms ------------------------------------------------------------------
  // RMSNorm (Llama): x * w / sqrt(mean(x^2) + eps) — no mean subtraction.
  // LayerNorm (GPT-2): (x - mu) / sigma * w + b.
  // Both are elementwise after a row reduction, so there is no matmul to send
  // through the units, and both run in f64-accumulated JS — which IEEE requires
  // to be exactly rounded, so every device agrees.
  function rmsNorm(x, rows, C, w, eps) {
    const y = new Float32Array(rows * C);
    for (let r = 0; r < rows; r++) {
      let s = 0;
      for (let j = 0; j < C; j++) { const v = x[r * C + j]; s += v * v; }
      const inv = 1 / Math.sqrt(s / C + eps);
      for (let j = 0; j < C; j++) y[r * C + j] = x[r * C + j] * inv * w[j];
    }
    return y;
  }
  function layerNorm(x, rows, C, w, b, eps) {
    const y = new Float32Array(rows * C);
    for (let r = 0; r < rows; r++) {
      let mu = 0;
      for (let j = 0; j < C; j++) mu += x[r * C + j];
      mu /= C;
      let v = 0;
      for (let j = 0; j < C; j++) { const d = x[r * C + j] - mu; v += d * d; }
      const inv = 1 / Math.sqrt(v / C + eps);
      for (let j = 0; j < C; j++) y[r * C + j] = (x[r * C + j] - mu) * inv * w[j] + (b ? b[j] : 0);
    }
    return y;
  }
  function norm(spec, x, rows, C, w, b) {
    return spec.norm === "rms" ? rmsNorm(x, rows, C, w, spec.normEps)
                               : layerNorm(x, rows, C, w, b, spec.normEps);
  }

  // ---- activations ------------------------------------------------------------
  const silu = (v) => v / (1 + Math.exp(-v));
  // tanh-approximate GELU: what GPT-2 was trained with, so it is the correct
  // function here rather than an approximation of the erf form.
  function gelu(v) {
    return 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
  }

  // ---- RoPE -------------------------------------------------------------------
  // Rotary embeddings applied per head, in the half-split layout HF uses:
  // dims [0, hd/2) pair with [hd/2, hd). Positions are absolute, and because
  // there is no KV cache every token re-runs the whole window, so position i
  // is simply the row index.
  function ropeTables(hd, theta, maxT) {
    const half = hd >> 1;
    const cos = new Float32Array(maxT * half), sin = new Float32Array(maxT * half);
    for (let p = 0; p < maxT; p++)
      for (let i = 0; i < half; i++) {
        const f = p / Math.pow(theta, (2 * i) / hd);
        cos[p * half + i] = Math.cos(f);
        sin[p * half + i] = Math.sin(f);
      }
    return { cos, sin, half };
  }
  // x is rows x (nHeads*hd), rows = T; rotate each head in place.
  function applyRope(x, T, nHeads, hd, rope, posOffset) {
    const half = rope.half, stride = nHeads * hd;
    for (let t = 0; t < T; t++) {
      const p = (posOffset || 0) + t;
      for (let h = 0; h < nHeads; h++) {
        const o = t * stride + h * hd;
        for (let i = 0; i < half; i++) {
          const c = rope.cos[p * half + i], s = rope.sin[p * half + i];
          const a = x[o + i], b = x[o + half + i];
          x[o + i] = a * c - b * s;
          x[o + half + i] = b * c + a * s;
        }
      }
    }
  }

  // ---- verified matmul --------------------------------------------------------
  // X is m x k, W is k x n (row-major). Weights are stored k x n at load time
  // (see loadStage), so nothing is transposed per call.
  async function vmm(X, W, m, k, n, ctx, bias) {
    const out = await V.vgemmBlock(X, W, { m, k, n, batch: 1 }, ctx.L, ctx.bgemm, ctx.audit);
    if (bias) for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] += bias[j];
    return out;
  }

  // ---- one layer --------------------------------------------------------------
  // `span` is the chunked-prefill path. Absent, the layer is the verified
  // full-window GEMM (one scale set for the whole window). Present, this stage
  // keeps K and V for its own layers and attends the new rows to that cache.
  // Those two paths are not bit-identical: per-column scales move when the
  // matrix grows. Callers must not mix them in one comparison.
  async function layerForward(S, x, ly, span) {
    const { spec, ctx } = S, C = spec.hidden;
    const T = span ? (x.length / C) | 0 : S.T;
    const hd = spec.headDim, nH = spec.heads, nKV = spec.kvHeads;
    const qDim = nH * hd, kvDim = nKV * hd;

    // ---- attention
    const h1 = norm(spec, x, T, C, ly.nrm1, ly.nrm1b);
    let q, k, v;
    if (spec.qkvFused) {
      // GPT-2 packs q,k,v into one (C x 3C) projection — one GEMM, then split.
      const qkv = await vmm(h1, ly.Wqkv, T, C, 3 * C, ctx, ly.bqkv);
      q = new Float32Array(T * C); k = new Float32Array(T * C); v = new Float32Array(T * C);
      for (let t = 0; t < T; t++) {
        q.set(qkv.subarray(t * 3 * C, t * 3 * C + C), t * C);
        k.set(qkv.subarray(t * 3 * C + C, t * 3 * C + 2 * C), t * C);
        v.set(qkv.subarray(t * 3 * C + 2 * C, t * 3 * C + 3 * C), t * C);
      }
    } else {
      [q, k, v] = await Promise.all([
        vmm(h1, ly.Wq, T, C, qDim, ctx, ly.bq),
        vmm(h1, ly.Wk, T, C, kvDim, ctx, ly.bk),
        vmm(h1, ly.Wv, T, C, kvDim, ctx, ly.bv),
      ]);
    }
    const pos0 = span ? span.pos0 : 0;
    if (spec.rope) {
      const need = pos0 + T + 1;
      if (!S.rope || S.rope.cos.length < need * (hd >> 1))
        S.rope = ropeTables(hd, spec.ropeTheta, Math.max(need, spec.maxPos || need));
      applyRope(q, T, nH, hd, S.rope, pos0);
      applyRope(k, T, nKV, hd, S.rope, pos0);
    }

    // Grouped-query attention: several query heads share one kv head. Rather
    // than materialising repeated kv (memory this project does not have to
    // spare), the head index is mapped when reading.
    const group = nH / nKV;
    const scale = 1 / Math.sqrt(hd);
    const ctxOut = new Float32Array(T * qDim);
    let cache = null, n0 = 0, N = T;
    if (span) {
      if (!span.cache) span.cache = [];
      cache = span.cache[span.li] || (span.cache[span.li] = { k: null, v: null, n: 0 });
      n0 = cache.n;
      N = n0 + T;
      const nk = new Float32Array(N * kvDim), nv = new Float32Array(N * kvDim);
      if (cache.k) { nk.set(cache.k); nv.set(cache.v); }
      nk.set(k, n0 * kvDim); nv.set(v, n0 * kvDim);
      cache.k = nk; cache.v = nv; cache.n = N;
      k = nk; v = nv;
    }
    // Scores and context are the two attention GEMMs. They are small per head
    // (T x hd and T x T), and the fused kernels in webgpu.js expect the
    // trainer's uniform-head layout, so at general head geometry they are run
    // through the same verified block GEMM per head — every product still goes
    // through the units, and the CPU mirror and GPU kernel agree bit-for-bit.
    for (let h = 0; h < nH; h++) {
      const kvh = Math.floor(h / group);
      const qh = new Float32Array(T * hd), kh = new Float32Array(N * hd), vh = new Float32Array(N * hd);
      for (let t = 0; t < T; t++)
        qh.set(q.subarray(t * qDim + h * hd, t * qDim + h * hd + hd), t * hd);
      for (let t = 0; t < N; t++) {
        kh.set(k.subarray(t * kvDim + kvh * hd, t * kvDim + kvh * hd + hd), t * hd);
        vh.set(v.subarray(t * kvDim + kvh * hd, t * kvDim + kvh * hd + hd), t * hd);
      }
      const khT = TC.transpose(kh, N, hd);                       // hd x N
      const scores = await V.vgemmBlock(qh, khT, { m: T, k: hd, n: N, batch: 1 }, ctx.L, ctx.bgemm, ctx.audit);
      const a = new Float32Array(T * N);                          // causal softmax
      for (let i = 0; i < T; i++) {
        const see = n0 + i;                                       // this query's own position in the cache
        let mx = -Infinity;
        for (let j = 0; j <= see; j++) mx = Math.max(mx, scores[i * N + j] * scale);
        let z = 0;
        for (let j = 0; j <= see; j++) { const e = Math.exp(scores[i * N + j] * scale - mx); a[i * N + j] = e; z += e; }
        for (let j = 0; j <= see; j++) a[i * N + j] /= z;
      }
      const oh = await V.vgemmBlock(a, vh, { m: T, k: N, n: hd, batch: 1 }, ctx.L, ctx.bgemm, ctx.audit);
      for (let t = 0; t < T; t++) ctxOut.set(oh.subarray(t * hd, t * hd + hd), t * qDim + h * hd);
    }
    const attn = await vmm(ctxOut, ly.Wo, T, qDim, C, ctx, ly.bo);
    const x2 = new Float32Array(T * C);
    for (let i = 0; i < x2.length; i++) x2[i] = x[i] + attn[i];

    // ---- MLP
    const h2 = norm(spec, x2, T, C, ly.nrm2, ly.nrm2b);
    let hid;
    if (spec.gated) {
      // SwiGLU: down(silu(gate(x)) * up(x))
      const [g, u] = await Promise.all([
        vmm(h2, ly.Wgate, T, C, spec.inter, ctx, null),
        vmm(h2, ly.Wup, T, C, spec.inter, ctx, null),
      ]);
      hid = g;
      for (let i = 0; i < hid.length; i++) hid[i] = silu(hid[i]) * u[i];
    } else {
      hid = await vmm(h2, ly.Wfc, T, C, spec.inter, ctx, ly.bfc);
      for (let i = 0; i < hid.length; i++) hid[i] = gelu(hid[i]);
    }
    const down = await vmm(hid, ly.Wdown, T, spec.inter, C, ctx, ly.bdown);
    const out = new Float32Array(T * C);
    for (let i = 0; i < out.length; i++) out[i] = x2[i] + down[i];
    return out;
  }

  // ---- stage ------------------------------------------------------------------
  function makeStage(spec, st, w, ctx, T) {
    const S = { spec, st, w, ctx, T: T || Math.min(spec.maxPos, 128) };
    if (spec.rope) S.rope = ropeTables(spec.headDim, spec.ropeTheta, S.T + 1);
    return S;
  }

  function embed(S, ids) {
    const { spec } = S, C = spec.hidden, T = S.T;
    const x = new Float32Array(T * C);
    for (let i = 0; i < T; i++) {
      const id = ids[i];
      for (let j = 0; j < C; j++) x[i * C + j] = S.w.emb[id * C + j];
    }
    if (S.w.pos)                                    // GPT-2 learned positions
      for (let i = 0; i < T; i++)
        for (let j = 0; j < C; j++) x[i * C + j] += S.w.pos[i * C + j];
    return x;
  }

  async function runLayers(S, x) {
    // A non-finite activation used to quantize to a silent zero and keep
    // travelling. Train refuses it (NonFiniteError). A hop must too: the
    // next stage would otherwise agree on a wrong sum.
    V.assertFinite(x, "activation");
    for (const ly of S.w.layers) x = await layerForward(S, x, ly);
    V.assertFinite(x, "activation");
    return x;
  }

  // One chunk of a pipelined prefill. `span.cache` is owned by the caller and
  // lives for the whole prompt; each call appends this chunk's K and V.
  // `span.pos0` is the first absolute position in `x`.
  async function runLayersChunk(S, x, span) {
    V.assertFinite(x, "activation");
    const C = S.spec.hidden;
    const T = (x.length / C) | 0;
    if (T < 1 || T * C !== x.length) throw new Error("chunk length is not a whole number of rows");
    for (let li = 0; li < S.w.layers.length; li++) {
      span.li = li;
      x = await layerForward(S, x, S.w.layers[li], span);
    }
    V.assertFinite(x, "activation");
    return x;
  }

  function embedRows(S, ids, pos0) {
    const { spec } = S, C = spec.hidden, T = ids.length;
    const x = new Float32Array(T * C);
    for (let i = 0; i < T; i++) {
      const id = ids[i];
      for (let j = 0; j < C; j++) x[i * C + j] = S.w.emb[id * C + j];
      if (S.w.pos) {
        const p = pos0 + i;
        for (let j = 0; j < C; j++) x[i * C + j] += S.w.pos[p * C + j];
      }
    }
    return x;
  }

  // Only the last position's logits are needed to pick the next token, and at
  // a 150k-token vocabulary that turns the largest GEMM in the model into a
  // single row — the biggest single saving in the ring.
  async function readout(S, x) {
    const { spec, ctx } = S, C = spec.hidden, T = S.T;
    const y = norm(spec, x, T, C, S.w.nrmF, S.w.nrmFb);
    const last = y.subarray((T - 1) * C, T * C);
    return vmm(last, S.w.lmHead, 1, C, spec.vocab, ctx, null);
  }

  // Logits at several rows of one window, so one lap can score a draft span.
  // Row indexes are into the fixed context window the verified path uses.
  async function logitsAt(S, x, rows) {
    const { spec, ctx } = S, C = spec.hidden;
    const T = (x.length / C) | 0;
    const y = norm(spec, x, T, C, S.w.nrmF, S.w.nrmFb);
    const out = [];
    for (const r of rows) {
      if (r < 0 || r >= T) throw new Error("readout row outside the window");
      out.push(await vmm(y.subarray(r * C, (r + 1) * C), S.w.lmHead, 1, C, spec.vocab, ctx, null));
    }
    return out;
  }

  // Chunked prefill, then one new token per lap on the same KV. Both the
  // ring and the differential check call this so they compare like with like.
  // It is not the full-window path.
  async function generateChunked(stages, ids0, nNew, chunkSize, opts) {
    const ids = [...ids0];
    const spans = stages.map(() => ({ cache: [], pos0: 0, li: 0 }));
    const size = Math.max(1, chunkSize | 0);
    let last = null;
    for (let lo = 0; lo < ids.length; lo += size) {
      const piece = ids.slice(lo, Math.min(ids.length, lo + size));
      let x = embedRows(stages[0], piece, lo);
      for (let s = 0; s < stages.length; s++) {
        spans[s].pos0 = lo;
        x = await runLayersChunk(stages[s], x, spans[s]);
      }
      last = x;
    }
    for (let n = 0; n < nNew; n++) {
      const row = (last.length / stages[0].spec.hidden) - 1;
      const id = pickToken((await logitsAt(stages[0], last, [row]))[0], opts);
      ids.push(id);
      const pos = ids.length - 1;
      let x = embedRows(stages[0], [id], pos);
      for (let s = 0; s < stages.length; s++) {
        spans[s].pos0 = pos;
        x = await runLayersChunk(stages[s], x, spans[s]);
      }
      last = x;
    }
    return ids;
  }

  // Where, in a left-padded window of length T, the logits sit that propose
  // each draft token plus the one token after the first rejection.
  function draftRows(T, seqLen, draftLen) {
    const rows = [];
    const first = seqLen - draftLen - 1;
    for (let i = 0; i <= draftLen; i++) {
      const tok = first + i;
      if (tok < 0 || tok >= seqLen) break;
      rows.push(T - seqLen + tok);
    }
    return rows;
  }

  // ---- sampling ---------------------------------------------------------------
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function pickToken(logits, opts) {
    const o = opts || {}, temp = o.temperature ?? 0, vocab = logits.length;
    if (!temp) {
      let best = 0, bv = -Infinity;
      for (let j = 0; j < vocab; j++) if (logits[j] > bv) { bv = logits[j]; best = j; }
      return best;
    }
    const k = Math.min(o.topK || 40, vocab);
    const idx = Array.from({ length: vocab }, (_, i) => i).sort((a, b) => logits[b] - logits[a]).slice(0, k);
    let mx = -Infinity;
    for (const i of idx) mx = Math.max(mx, logits[i] / temp);
    let z = 0;
    const p = idx.map(i => { const e = Math.exp(logits[i] / temp - mx); z += e; return e; });
    let r = (o.rng || Math.random)() * z;
    for (let i = 0; i < idx.length; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
    return idx[idx.length - 1];
  }

  // ---- single-device reference ------------------------------------------------
  async function generateLocal(stages, ids, nTokens, opts) {
    const head = stages[0], T = head.T;
    const out = [...ids];
    for (let n = 0; n < nTokens; n++) {
      const win = new Int32Array(T);
      const tail = out.slice(-T);
      for (let i = 0; i < tail.length; i++) win[T - tail.length + i] = tail[i];
      let x = embed(head, win);
      for (const S of stages) x = await runLayers(S, x);
      out.push(pickToken(await readout(head, x), opts));
    }
    return out;
  }

  // Verified full-window path, with a prompt-lookup draft scored in the same lap.
  // An empty draft is exactly one new token, the same window as generateLocal.
  async function generateLookahead(stages, ids0, nNew, T, opts) {
    const ids = [...ids0];
    const head = stages[0];
    const maxDraft = opts && opts.maxDraft != null ? opts.maxDraft : 4;
    let produced = 0;
    while (produced < nNew) {
      let draft = SH.promptLookupDraft(ids, maxDraft);
      const seqIds = ids.concat(draft);
      const shown = seqIds.slice(-T);
      const shownDraft = Math.min(draft.length, Math.max(0, shown.length - 1));
      draft = draft.slice(draft.length - shownDraft);
      const win = new Int32Array(T);
      for (let i = 0; i < shown.length; i++) win[T - shown.length + i] = shown[i];
      let x = embed(head, win);
      for (const S of stages) x = await runLayers(S, x);
      const rows = draftRows(T, shown.length, draft.length);
      const picked = [];
      for (const logits of await logitsAt(head, x, rows)) picked.push(pickToken(logits, opts));
      const got = SH.acceptDraft(picked, draft);
      if (!got.length) break;
      for (const id of got) {
        ids.push(id);
        produced++;
        if (produced >= nNew) break;
      }
    }
    return ids;
  }

  const api = { makeStage, embed, embedRows, runLayers, runLayersChunk, readout, logitsAt, draftRows,
                generateLookahead,
                pickToken, generateLocal, generateChunked,
                rmsNorm, layerNorm, norm, ropeTables, applyRope, silu, gelu, mulberry32 };
  if (typeof module !== "undefined" && module.exports) {
    TC = require("./traincore.js"); V = require("./verified_core.js"); SH = require("./shard.js");
    module.exports = api;
  } else { TC = root.TrainCore; V = root.Verified; SH = root.Shard; root.Infer = api; }
})(typeof self !== "undefined" ? self : this);
