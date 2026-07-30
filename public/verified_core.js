// Verified INT8 compute — the emulated GPU logic, in the browser.
// A layer's forward runs THROUGH the units: quantize -> LUT multiply -> requant
// -> optional ReLU -> dequant. Backward is a straight-through estimator (the
// integer path has no gradient), so ordinary float weights still learn.
// Same units as the Python/Docker DaisyChain; here they're lookup tables.
(function (root) {
  "use strict";

  let TC;   // TrainCore (matmul/transpose) — resolved per environment at the end

  function quantize(X) {
    let mx = 0; for (let i = 0; i < X.length; i++) { const a = Math.abs(X[i]); if (a > mx) mx = a; }
    const scale = Math.max(mx / 127, 1e-8);
    const q = new Int8Array(X.length);
    for (let i = 0; i < X.length; i++) { let v = Math.round(X[i] / scale); q[i] = v < -128 ? -128 : v > 127 ? 127 : v; }
    return { q, scale };
  }

  // int8 matmul via the verified multiply LUT: acc(m×n) = sum_k mulLUT[Xq,Wq]
  function lutMatmulJS(Xq, Wq, m, k, n, L) {
    const C = new Int32Array(m * n), mul = L.mul;
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const au = (Xq[i * k + p] & 0xFF) * 256, wo = p * n, co = i * n;
        for (let j = 0; j < n; j++) C[co + j] += mul[au + (Wq[wo + j] & 0xFF)];
      }
    }
    return C;
  }

  // ---- 3xINT8 fast-accurate GEMM --------------------------------------------
  // The CUTLASS example-27 "3xTF32" scheme, ported to the verified units:
  // split each float into a coarse int8 part plus an int8-quantized residual,
  // run three EXACT LUT GEMMs (hi·hi, hi·lo, lo·hi), drop the negligible
  // lo·lo, and recombine. Same big/small decomposition NVIDIA uses to recover
  // near-fp32 accuracy from TF32 tensor cores — here it recovers ~14-bit
  // accuracy from the 8-bit units, at 3× the unit ops. Every product still
  // goes through the verified mul8 LUT.
  function quantize2(X) {
    const hi = quantize(X);
    const r = new Float32Array(X.length);
    for (let i = 0; i < X.length; i++) r[i] = X[i] - hi.q[i] * hi.scale;
    const lo = quantize(r);
    return { hi, lo };
  }
  function combine3(hh, hl, lh, x, w, len) {
    const out = new Float32Array(len);
    const shh = x.hi.scale * w.hi.scale, shl = x.hi.scale * w.lo.scale, slh = x.lo.scale * w.hi.scale;
    for (let i = 0; i < len; i++) out[i] = hh[i] * shh + hl[i] * shl + lh[i] * slh;
    return out;
  }
  function lutMatmul3JS(Xf, Wf, m, k, n, L) {          // sync, CPU LUT path
    const x = quantize2(Xf), w = quantize2(Wf);
    return combine3(lutMatmulJS(x.hi.q, w.hi.q, m, k, n, L),
                    lutMatmulJS(x.hi.q, w.lo.q, m, k, n, L),
                    lutMatmulJS(x.lo.q, w.hi.q, m, k, n, L), x, w, m * n);
  }
  async function lutMatmul3(Xf, Wf, m, k, n, L, matmulInt8) {  // any backend
    const x = quantize2(Xf), w = quantize2(Wf);
    const mm = matmulInt8 || lutMatmulJS;
    const [hh, hl, lh] = await Promise.all([
      mm(x.hi.q, w.hi.q, m, k, n, L),
      mm(x.hi.q, w.lo.q, m, k, n, L),
      mm(x.lo.q, w.hi.q, m, k, n, L),
    ]);
    return combine3(hh, hl, lh, x, w, m * n);
  }

  // ---- block-scaled verified GEMM (CUTLASS ex. 67/81 blockwise scaling) ------
  // Per-ROW scales for the activations and per-COLUMN scales for the weights:
  // the integer math through the mul8 LUT is completely unchanged — only the
  // dequant uses rs[row]·cs[col] instead of one tensor-wide product, so a single
  // outlier no longer crushes the quantization resolution of every other
  // row/column. One LUT pass at this granularity beats the per-tensor 3-pass.
  function quantizeRows(X, rows, cols) {
    const q = new Int8Array(rows * cols), s = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let mx = 0;
      for (let c = 0; c < cols; c++) { const a = Math.abs(X[r * cols + c]); if (a > mx) mx = a; }
      const sc = Math.max(mx / 127, 1e-8); s[r] = sc;
      for (let c = 0; c < cols; c++) {
        const v = Math.round(X[r * cols + c] / sc);
        q[r * cols + c] = v < -128 ? -128 : v > 127 ? 127 : v;
      }
    }
    return { q, s };
  }
  function quantizeCols(W, rows, cols) {
    const q = new Int8Array(rows * cols), s = new Float32Array(cols);
    for (let c = 0; c < cols; c++) {
      let mx = 0;
      for (let r = 0; r < rows; r++) { const a = Math.abs(W[r * cols + c]); if (a > mx) mx = a; }
      s[c] = Math.max(mx / 127, 1e-8);
    }
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = Math.round(W[r * cols + c] / s[c]);
        q[r * cols + c] = v < -128 ? -128 : v > 127 ? 127 : v;
      }
    return { q, s };
  }

  // ---- B2B MLP chain (CUTLASS ex. 13 two-GEMM fusion + ex. 23 epilogue
  // reduction), respecced for cross-device exactness ---------------------------
  // The MLP is the one back-to-back GEMM pair with no layernorm/softmax between
  // (ReLU is already fused in the epilogue), so the intermediate h1 can be
  // quantized ON the GPU and fed straight to the second GEMM. Two rules make
  // that fleet-safe:
  //   1. The per-row |max| (ex. 23) uses only comparisons — exact on any
  //      hardware, order-independent — and comes back to JS as ~1KB.
  //   2. Scale DERIVATION (two divisions) stays in JS f64, which IEEE requires
  //      to be exactly rounded and is therefore identical on every device.
  //      WGSL division is only 2.5 ULP — a fork waiting to happen — but WGSL
  //      multiply/add are correctly rounded and floor/clamp are exact. So the
  //      quantize step is respecced from  round(x / scale)  to
  //      floor(f32(x * invScale) + 0.5)  — floor(x+0.5) IS Math.round's tie
  //      rule — and the fround-stepped mirror below is bit-identical to the
  //      GPU kernel, which is exact-gated against it at init.
  // NOTE this changes which int8 a value on a rounding boundary lands on
  // (≤1 step) vs quantizeRows, so old and new builds cannot co-train — the
  // divergence guard stops such mixed groups by design.
  function rowAbsMax(X, rows, cols) {
    const mx = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let m = 0;
      for (let c = 0; c < cols; c++) { const a = Math.abs(X[r * cols + c]); if (a > m) m = a; }
      mx[r] = m;
    }
    return mx;
  }
  function scalesFromAbsMax(mx) {                // f64 divisions: exactly rounded, device-identical
    const scale = new Float32Array(mx.length), inv = new Float32Array(mx.length);
    for (let i = 0; i < mx.length; i++) {
      scale[i] = Math.max(mx[i] / 127, 1e-8);
      inv[i] = 1 / scale[i];                     // recip of the STORED f32 scale
    }
    return { scale, inv };
  }
  function quantizeRowsInv(X, rows, cols, inv) { // bit-exact mirror of the GPU quantize kernel
    const q = new Int8Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const iv = inv[r];
      for (let c = 0; c < cols; c++) {
        const n = Math.floor(f32(f32(X[r * cols + c] * iv) + 0.5));
        q[r * cols + c] = n < -128 ? -128 : n > 127 ? 127 : n;
      }
    }
    return q;
  }
  // the chained MLP: X @ W1 -> ReLU (fused) -> absmax -> quantize -> @ W2.
  //   d = { m, k, h, n };  gpuMlp (from webgpu.js) runs both GEMMs + the
  // on-GPU quantize with one tiny absmax readback between; without it the CPU
  // mirror chain runs — SAME math, so mixed GPU/CPU fleets stay bit-identical.
  async function vmlpBlock(Xf, W1f, W2f, d, L, gpuMlp, audit) {
    const x = quantizeRows(Xf, d.m, d.k);
    const w1 = quantizeCols(W1f, d.k, d.h);
    const w2 = quantizeCols(W2f, d.h, d.n);
    if (gpuMlp) {
      const r = await gpuMlp(x.q, w1.q, w2.q, x.s, w1.s, w2.s, d);
      if (audit && audit.due()) {
        // audit BOTH live GEMMs: gemm1 against the units directly; gemm2 by
        // reconstructing its exact operand through the proven quantize mirror
        const bad1 = auditTile(x.q, w1.q, x.s, w1.s, { m: d.m, k: d.k, n: d.h, relu: true }, r.h1, L, audit.cells);
        if (bad1) { audit.fail("mlp gemm1: " + bad1); return r; }
        const sc = scalesFromAbsMax(rowAbsMax(r.h1, d.m, d.h));
        const hq = quantizeRowsInv(r.h1, d.m, d.h, sc.inv);
        const bad2 = auditTile(hq, w2.q, sc.scale, w2.s, { m: d.m, k: d.h, n: d.n }, r.out, L, audit.cells);
        if (bad2) audit.fail("mlp gemm2: " + bad2);
      }
      return r;
    }
    const h1 = bgemmJS(x.q, w1.q, x.s, w1.s, { m: d.m, k: d.k, n: d.h, batch: 1, relu: true }, L);
    const sc = scalesFromAbsMax(rowAbsMax(h1, d.m, d.h));
    const hq = quantizeRowsInv(h1, d.m, d.h, sc.inv);
    const out = bgemmJS(hq, w2.q, sc.scale, w2.s, { m: d.m, k: d.h, n: d.n, batch: 1 }, L);
    return { h1, out };
  }

  // ---- epilogue mirror -------------------------------------------------------
  // BIT-EXACT mirror of the WGSL epilogue `f32(s) * a * b`. WGSL rounds to f32
  // after the int->float conversion and after EACH multiply; plain JS would do
  // the whole chain in f64 and round once, which differs in the last ulp. That
  // last-ulp gap is what used to force a tolerance into the kernel gates —
  // mirroring the rounding exactly is what lets the gates compare with `!==`.
  const f32 = Math.fround;
  function epi(s, a, b) { return f32(f32(f32(s) * a) * b); }

  // f32 equality at the BIT level: `!==` says -0 === 0, but replicas are
  // compared by hashing raw bytes, so an audit that can't see the sign of zero
  // could pass a device that later forks the fleet. (Real ISAs have non-IEEE
  // modes that flush -0 to +0 — e.g. RDNA2 output modifiers / legacy muls.)
  const _fb = new Float32Array(1), _ub = new Uint32Array(_fb.buffer);
  function bitDiff(a, b) { _fb[0] = a; const u = _ub[0]; _fb[0] = b; return u !== _ub[0]; }

  // CPU mirror of the fused GPU kernel: batched int8 GEMM through the LUT with
  // the epilogue (block dequant + optional ReLU) applied before returning —
  // exactly what the WGSL kernel does on-device. d.acc=true returns the raw
  // int32 accumulator instead (the exact oracle the fused kernel normally hides).
  function bgemmJS(Xq, Wq, rs, cs, d, L) {
    const { m, k, n } = d, batch = d.batch || 1, relu = !!d.relu, mul = L.mul;
    const raw = !!d.acc;
    const out = raw ? new Int32Array(batch * m * n) : new Float32Array(batch * m * n);
    const acc = new Int32Array(n);
    for (let bz = 0; bz < batch; bz++) {
      const xo = bz * m * k, wo = bz * k * n, oo = bz * m * n, co = bz * n;
      for (let i = 0; i < m; i++) {
        acc.fill(0);
        const xrow = xo + i * k;
        for (let p = 0; p < k; p++) {
          const au = (Xq[xrow + p] & 0xFF) * 256, wrow = wo + p * n;
          for (let j = 0; j < n; j++) acc[j] += mul[au + (Wq[wrow + j] & 0xFF)];
        }
        const orow = oo + i * n;
        if (raw) { for (let j = 0; j < n; j++) out[orow + j] = acc[j]; continue; }
        const rscale = rs[bz * m + i];
        for (let j = 0; j < n; j++) {
          const v = epi(acc[j], rscale, cs[co + j]);
          out[orow + j] = relu && v < 0 ? 0 : v;
        }
      }
    }
    return out;
  }

  // Recompute a handful of RANDOM output cells of a live GEMM through the LUT
  // mirror and compare against what the kernel produced. Sampling cells instead
  // of whole matrices makes this cheap enough to run continuously, at the real
  // shapes training uses — not once at boot on toy inputs.
  // STRATIFIED sampling. Uniformly random cells are the wrong instrument for
  // the bugs that actually occur here: a bounds-guard off-by-one or a pack-tail
  // padding bug lives on the LAST row/column, and uniform sampling finds that
  // with probability ~1/n per cell — at the 16512-wide logits GEMM, never. So
  // the first cells are the structurally dangerous ones (corners, last row,
  // last column, last batch) chosen deterministically, and the remainder are
  // random interior cells that catch diffuse bugs. Same principle as poisoning
  // the buffer pool: construct the dangerous case, don't wait to land on it.
  function auditTile(Xq, Wq, rs, cs, d, got, L, nCells) {
    const { m, k, n } = d, batch = d.batch || 1, relu = !!d.relu, mul = L.mul;
    const N = nCells || 8;
    const edges = [[0, m - 1, n - 1], [0, 0, n - 1], [0, m - 1, 0], [0, 0, 0],
                   [batch - 1, m - 1, n - 1], [batch - 1, 0, 0]];
    for (let t = 0; t < N; t++) {
      let bz, i, j;
      if (t < edges.length) { bz = edges[t][0]; i = edges[t][1]; j = edges[t][2]; }
      else { bz = (Math.random() * batch) | 0; i = (Math.random() * m) | 0; j = (Math.random() * n) | 0; }
      let acc = 0;
      const xrow = bz * m * k + i * k, wo = bz * k * n;
      for (let p = 0; p < k; p++) acc += mul[(Xq[xrow + p] & 0xFF) * 256 + (Wq[wo + p * n + j] & 0xFF)];
      let v = epi(acc, rs[bz * m + i], cs[bz * n + j]);
      if (relu && v < 0) v = 0;
      const idx = (bz * m + i) * n + j;
      if (bitDiff(got[idx], v))
        return `GEMM audit failed at [b${bz},${i},${j}] shape ${m}x${k}x${n}: kernel ${Object.is(got[idx], -0) ? "-0" : got[idx]} vs units ${Object.is(v, -0) ? "-0" : v}`;
    }
    return null;
  }

  // ---- exact mirror of the split-K f32 GEMM ----------------------------------
  // The f32 backward GEMM was the last kernel gated by a TOLERANCE (allclose at
  // 1e-3) — and this project's own gate mutation test shows allclose waving
  // through real bugs. The reason was real though: split-K accumulates in a
  // different ORDER than a naive reference, so bit-equality against the naive
  // one is impossible. The fix is the same as the epilogue mirror: reproduce
  // the kernel's order exactly, then compare with `!==`.
  //   partials: for z in 0..S-1, sum p in [z*ks, min(k,(z+1)*ks)) in order
  //   reduce:   sum the S partials in ascending z
  // `fma` selects the rounding schedule for `s + a*b`: WGSL PERMITS a compiler
  // to contract that into a fused multiply-add (one rounding) instead of two.
  // Which one the device does is a fact about the device, so the gate tries
  // both and reports which matches rather than assuming.
  function fgemmMirror(A, Bm, d, fma) {
    const { m, k, n } = d, transA = !!d.transA;
    const S = k > 4096 ? Math.min(16, Math.ceil(k / 2048)) : 1;
    const ks = Math.ceil(k / S);
    const out = new Float32Array(m * n);
    for (let row = 0; row < m; row++)
      for (let col = 0; col < n; col++) {
        let acc = 0;                                    // reduce pass, ascending z
        for (let z = 0; z < S; z++) {
          const p0 = z * ks, p1 = Math.min(k, p0 + ks);
          let s = 0;                                    // one partial, in order
          for (let p = p0; p < p1; p++) {
            const a = transA ? A[p * m + row] : A[row * k + p];
            s = fma ? f32(s + a * Bm[p * n + col])      // single rounding
                    : f32(s + f32(a * Bm[p * n + col]));
          }
          acc = f32(acc + s);
        }
        out[row * n + col] = acc;
      }
    return out;
  }

  // ---- live audits for the fused attention kernels ---------------------------
  // The attention kernels had exact INIT gates but nothing at live shapes —
  // the exact gap the GEMM audit exists to close, left open on the kernels with
  // the trickiest indexing (head-strided gather, scatter write-back). These
  // recompute individual output cells from the units, stratified like
  // auditTile: last/first token pair, last head, last channel first, then
  // random. Cost is hd (or T) multiply-adds per cell.
  function auditAttScores(qq, kq, qs, ks, d, got, L, nCells) {
    const { B, T, heads, hd } = d, C = heads * hd, mul = L.mul, raw = !!d.acc;
    const N = nCells || 8;
    const edges = [[B - 1, heads - 1, T - 1, T - 1], [0, 0, 0, 0],
                   [0, heads - 1, T - 1, 0], [B - 1, 0, 0, T - 1]];
    for (let t = 0; t < N; t++) {
      let bi, h, ti, tj;
      if (t < edges.length) { bi = edges[t][0]; h = edges[t][1]; ti = edges[t][2]; tj = edges[t][3]; }
      else { bi = (Math.random() * B) | 0; h = (Math.random() * heads) | 0;
             ti = (Math.random() * T) | 0; tj = (Math.random() * T) | 0; }
      const bz = bi * heads + h;
      const qo = (bi * T + ti) * C + h * hd, ko = (bi * T + tj) * C + h * hd;
      let acc = 0;
      for (let p = 0; p < hd; p++) acc += mul[(qq[qo + p] & 0xFF) * 256 + (kq[ko + p] & 0xFF)];
      const v = raw ? acc : epi(acc, qs[(bi * T + ti) * heads + h], ks[(bi * T + tj) * heads + h]);
      const idx = (bz * T + ti) * T + tj;
      if (raw ? got[idx] !== v : bitDiff(got[idx], v))
        return `att.scores audit failed at [b${bi},h${h},${ti},${tj}] B${B}T${T}H${heads}d${hd}: kernel ${got[idx]} vs units ${v}`;
    }
    return null;
  }
  function auditAttCtx(aq, vq, as, vs, d, got, L, nCells) {
    const { B, T, heads, hd } = d, C = heads * hd, mul = L.mul, raw = !!d.acc;
    const N = nCells || 8;
    const edges = [[B - 1, heads - 1, T - 1, hd - 1], [0, 0, 0, 0],
                   [0, heads - 1, T - 1, 0], [B - 1, 0, 0, hd - 1]];
    for (let t = 0; t < N; t++) {
      let bi, h, ti, j;
      if (t < edges.length) { bi = edges[t][0]; h = edges[t][1]; ti = edges[t][2]; j = edges[t][3]; }
      else { bi = (Math.random() * B) | 0; h = (Math.random() * heads) | 0;
             ti = (Math.random() * T) | 0; j = (Math.random() * hd) | 0; }
      const bz = bi * heads + h, ao = (bz * T + ti) * T;
      let acc = 0;
      for (let tj = 0; tj < T; tj++)
        acc += mul[(aq[ao + tj] & 0xFF) * 256 + (vq[(bi * T + tj) * C + h * hd + j] & 0xFF)];
      const v = raw ? acc : epi(acc, as[bz * T + ti], vs[(bi * heads + h) * hd + j]);
      const idx = (bi * T + ti) * C + h * hd + j;
      if (raw ? got[idx] !== v : bitDiff(got[idx], v))
        return `att.ctx audit failed at [b${bi},h${h},${ti},${j}] B${B}T${T}H${heads}d${hd}: kernel ${got[idx]} vs units ${v}`;
    }
    return null;
  }

  // block-scaled verified GEMM, float in → float out.
  //   d = { m, k, n, batch=1, relu=false }; X is (batch·m)×k, W is batch×(k×n)
  // gpuBgemm (from webgpu.js) runs the batched kernel with the fused epilogue;
  // without it the CPU LUT mirror runs. Every product goes through the units.
  async function vgemmBlock(Xf, Wf, d, L, gpuBgemm, audit) {
    const { m, k, n } = d, batch = d.batch || 1;
    const x = quantizeRows(Xf, batch * m, k);
    let wq, ws;
    if (batch === 1) {
      const w = quantizeCols(Wf, k, n); wq = w.q; ws = w.s;
    } else {
      wq = new Int8Array(batch * k * n); ws = new Float32Array(batch * n);
      for (let bz = 0; bz < batch; bz++) {
        const w = quantizeCols(Wf.subarray(bz * k * n, (bz + 1) * k * n), k, n);
        wq.set(w.q, bz * k * n); ws.set(w.s, bz * n);
      }
    }
    if (gpuBgemm) {
      const out = await gpuBgemm(x.q, wq, x.s, ws, d);
      // continuous re-verification at LIVE shapes: the boot gate only ever saw
      // toy inputs, so sample a few real cells against the units as we go
      if (audit && audit.due()) {
        const bad = auditTile(x.q, wq, x.s, ws, d, out, L, audit.cells);
        if (bad) audit.fail(bad);
      }
      return out;
    }
    return bgemmJS(x.q, wq, x.s, ws, d, L);
  }

  // ---- gather-fused attention through the units (CUTLASS ex. 36/52) ----------
  // The kernels read q/k/v/ctx directly in their natural BT×C layout with
  // head-strided indexing — no JS gather copies, no kᵀ transpose, and the
  // context write scatters straight back into BT×C. Quantization stays
  // block-scaled: q/k/a per (token,head) row, v per (head,channel) column.
  // The (BT·heads)×hd row view of q/k IS the contiguous buffer, so
  // quantizeRows(q, BT·heads, hd) gives per-(token,head) scales for free.
  function quantizeHeadCols(v, B, T, heads, hd) {   // per (batch,head,channel) column
    const C = heads * hd;
    const q = new Int8Array(B * T * C), s = new Float32Array(B * heads * hd);
    for (let bi = 0; bi < B; bi++)
      for (let h = 0; h < heads; h++)
        for (let j = 0; j < hd; j++) {
          let mx = 0;
          for (let ti = 0; ti < T; ti++) {
            const a = Math.abs(v[(bi * T + ti) * C + h * hd + j]);
            if (a > mx) mx = a;
          }
          const sc = Math.max(mx / 127, 1e-8);
          s[(bi * heads + h) * hd + j] = sc;
          for (let ti = 0; ti < T; ti++) {
            const idx = (bi * T + ti) * C + h * hd + j;
            const w = Math.round(v[idx] / sc);
            q[idx] = w < -128 ? -128 : w > 127 ? 127 : w;
          }
        }
    return { q, s };
  }
  // scores S[bz,ti,tj] = q_row(bi,ti,h) · k_row(bi,tj,h), every product via the LUT
  // d.acc=true returns the raw int32 accumulator (exact oracle for the kernel gate)
  function attScoresJS(qq, kq, qs, ks, d, L) {
    const { B, T, heads, hd } = d, C = heads * hd, mul = L.mul, raw = !!d.acc;
    const out = raw ? new Int32Array(B * heads * T * T) : new Float32Array(B * heads * T * T);
    for (let bi = 0; bi < B; bi++) for (let h = 0; h < heads; h++) {
      const bz = bi * heads + h;
      for (let ti = 0; ti < T; ti++) {
        const qo = (bi * T + ti) * C + h * hd, rscale = qs[(bi * T + ti) * heads + h];
        for (let tj = 0; tj < T; tj++) {
          const ko = (bi * T + tj) * C + h * hd;
          let acc = 0;
          for (let p = 0; p < hd; p++) acc += mul[(qq[qo + p] & 0xFF) * 256 + (kq[ko + p] & 0xFF)];
          out[(bz * T + ti) * T + tj] = raw ? acc : epi(acc, rscale, ks[(bi * T + tj) * heads + h]);
        }
      }
    }
    return out;
  }
  // ctx[(bi,ti),(h,j)] = Σ_tj a[bz,ti,tj]·v[(bi,tj),(h,j)] — scatter fused into BT×C
  function attCtxJS(aq, vq, as, vs, d, L) {
    const { B, T, heads, hd } = d, C = heads * hd, mul = L.mul, raw = !!d.acc;
    const out = raw ? new Int32Array(B * T * C) : new Float32Array(B * T * C);
    for (let bi = 0; bi < B; bi++) for (let h = 0; h < heads; h++) {
      const bz = bi * heads + h;
      for (let ti = 0; ti < T; ti++) {
        const ao = (bz * T + ti) * T, rscale = as[bz * T + ti];
        for (let j = 0; j < hd; j++) {
          let acc = 0;
          for (let tj = 0; tj < T; tj++)
            acc += mul[(aq[ao + tj] & 0xFF) * 256 + (vq[(bi * T + tj) * C + h * hd + j] & 0xFF)];
          out[(bi * T + ti) * C + h * hd + j] = raw ? acc : epi(acc, rscale, vs[(bi * heads + h) * hd + j]);
        }
      }
    }
    return out;
  }

  // one verified layer forward; returns float out (+ cache for STE backward).
  // Every product goes through the verified INT8 multiply (mul8 LUT) with exact
  // int32 accumulation — i.e. an emulated INT8 tensor-core GEMM — then dequant.
  async function linearFwd(X, W, m, k, n, L, useRelu, matmulInt8) {
    const xq = quantize(X), wq = quantize(W);
    const acc = await (matmulInt8 || lutMatmulJS)(xq.q, wq.q, m, k, n, L);  // verified multiply
    const dq = xq.scale * wq.scale;
    const out = new Float32Array(m * n);
    const mask = useRelu ? new Uint8Array(m * n) : null;
    for (let i = 0; i < m * n; i++) {
      let v = acc[i] * dq;
      if (useRelu) { if (v > 0) mask[i] = 1; else v = 0; }
      out[i] = v;
    }
    return { out, mask };
  }

  // 2-layer MLP: X→H (relu) →dout. Forward through verified units, MSE loss.
  async function forward(X, y, W1, W2, D, L, matmulInt8) {
    const { n, din, h, dout } = D;
    const l1 = await linearFwd(X, W1, n, din, h, L, true, matmulInt8);
    const l2 = await linearFwd(l1.out, W2, n, h, dout, L, false, matmulInt8);
    const resid = new Float32Array(n * dout); let loss = 0;
    for (let i = 0; i < resid.length; i++) { const r = l2.out[i] - y[i]; resid[i] = r; loss += r * r; }
    loss /= resid.length;
    return { loss, resid, z1: l1.out, mask1: l1.mask };
  }

  // STE backward (verified matmul treated as float X@W). Returns flat [gW1, gW2].
  function backward(X, W1, W2, fwd, D) {
    const { n, din, h, dout } = D;
    const { resid, z1, mask1 } = fwd;
    const s = 2 / n;
    const dout_ = new Float32Array(resid.length);
    for (let i = 0; i < resid.length; i++) dout_[i] = resid[i] * s;
    const mm = TC.matmul, tr = TC.transpose;
    const gW2 = mm(tr(z1, n, h), dout_, h, n, dout);          // z1ᵀ @ dout
    const dz1 = mm(dout_, tr(W2, h, dout), n, dout, h);        // dout @ W2ᵀ
    for (let i = 0; i < dz1.length; i++) if (!mask1[i]) dz1[i] = 0;  // relu grad
    const gW1 = mm(tr(X, n, din), dz1, din, n, h);            // Xᵀ @ dz1
    const g = new Float32Array(gW1.length + gW2.length);
    g.set(gW1, 0); g.set(gW2, gW1.length);
    return g;
  }

  function splitApply(W1, W2, gAvg, lr) {
    for (let i = 0; i < W1.length; i++) W1[i] -= lr * gAvg[i];
    for (let j = 0; j < W2.length; j++) W2[j] -= lr * gAvg[W1.length + j];
  }

  const api = { quantize, quantize2, quantizeRows, quantizeCols, quantizeHeadCols, lutMatmulJS, lutMatmul3JS, lutMatmul3,
                bgemmJS, vgemmBlock, auditTile, epi, attScoresJS, attCtxJS, linearFwd, forward, backward, splitApply,
                rowAbsMax, scalesFromAbsMax, quantizeRowsInv, vmlpBlock, bitDiff,
                auditAttScores, auditAttCtx, fgemmMirror };
  if (typeof module !== "undefined" && module.exports) { TC = require("./traincore.js"); module.exports = api; }
  else { TC = root.TrainCore; root.Verified = api; }
})(typeof self !== "undefined" ? self : this);
