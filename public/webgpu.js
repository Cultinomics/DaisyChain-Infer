// WebGPU INT8 matmul via the verified multiply LUT — the emulated GPU logic
// running on the browser's GPU. Automatic CPU fallback (same LUT) for machines
// without WebGPU (e.g. old PCs via Supermium). initCompute() returns
//   { backend, label, matmulInt8(Xq, Wq, m, k, n, L) -> Int32Array }
// matching Verified.lutMatmulJS, so the trainer is device-blind.
//
// High-throughput path: when the browser exposes WGSL's
// packed_4x8_integer_dot_product feature, we use dot4I8Packed — it compiles to
// the GPU's DP4A/INT8 dot-product hardware (the same units tensor-core INT8
// paths are built on): 4 exact int8 MACs per instruction, int32 accumulation,
// and 4× less memory traffic from packing. Because int8×int8→int32 is exact,
// it is bit-identical to the verified mul8 LUT — and we PROVE that at init by
// cross-checking random matmuls against the LUT before trusting it. If the
// hardware ever disagrees with the units, we fall back to the LUT shader.
(function (root) {
  "use strict";

  const WGSL_LUT = `
    @group(0) @binding(0) var<storage, read> Xq  : array<i32>;   // int8 byte per elem
    @group(0) @binding(1) var<storage, read> Wq  : array<i32>;
    @group(0) @binding(2) var<storage, read> lut : array<i32>;   // 65536 signed products
    @group(0) @binding(3) var<storage, read_write> C : array<i32>;
    @group(0) @binding(4) var<uniform> dims : vec3<u32>;         // m, k, n
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let k = dims.y; let n = dims.z;
      let row = gid.x; let col = gid.y;
      if (row >= m || col >= n) { return; }
      var s : i32 = 0;
      for (var p = 0u; p < k; p = p + 1u) {
        let au = u32(Xq[row * k + p] & 255);
        let bu = u32(Wq[p * n + col] & 255);
        s = s + lut[au * 256u + bu];
      }
      C[row * n + col] = s;
    }`;

  // ---- B2B MLP chain kernels (CUTLASS ex. 13 + 23) ---------------------------
  // ROWMAX: per-row |max| of a GEMM's f32 output, fused into the same command
  // encoder (ex. 23 epilogue reduction). Non-negative f32 bit patterns order
  // like u32, so atomicMax on bitcast(abs(v)) computes an EXACT max, in any
  // execution order, on any hardware — nothing here can round.
  const WGSL_ROWMAX = `
    @group(0) @binding(0) var<storage, read> O : array<f32>;
    @group(0) @binding(1) var<storage, read_write> MX : array<atomic<u32>>;
    @group(0) @binding(2) var<uniform> dims : vec4<u32>;         // m, n, _, _
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let n = dims.y;
      let row = gid.x; let col = gid.y;
      if (row >= m || col >= n) { return; }
      atomicMax(&MX[row], bitcast<u32>(abs(O[row * n + col])));
    }`;
  // QUANT: h1 (f32, still on the GPU) -> int8 by MULTIPLY with a JS-computed
  // inverse scale. floor(f32(x*inv)+0.5) uses only ops WGSL guarantees exact
  // or correctly rounded (mul, add, floor, clamp) — division is 2.5 ULP and
  // never runs on the GPU. Bit-identical to Verified.quantizeRowsInv, and
  // exact-gated against it at init. pack=true emits 4 bytes per u32 for the
  // DP4A kernel; pack=false emits one i32 per element for the LUT kernel.
  const WGSL_QUANT = (pack) => `
    @group(0) @binding(0) var<storage, read> H : array<f32>;
    @group(0) @binding(1) var<storage, read> inv : array<f32>;   // per row
    @group(0) @binding(2) var<storage, read_write> Q : array<${pack ? "u32" : "i32"}>;
    @group(0) @binding(3) var<uniform> dims : vec4<u32>;         // m, k, kw, _
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let k = dims.y; let kw = dims.z;
      let idx = gid.x;
      ${pack ? `
      if (idx >= m * kw) { return; }
      let row = idx / kw;
      var acc : u32 = 0u;
      for (var b = 0u; b < 4u; b = b + 1u) {
        let c = (idx % kw) * 4u + b;
        var q : i32 = 0;
        if (c < k) {
          let v = clamp(floor(H[row * k + c] * inv[row] + 0.5), -128.0, 127.0);
          q = i32(v);
        }
        acc = acc | ((u32(q) & 255u) << (8u * b));
      }
      Q[idx] = acc;` : `
      if (idx >= m * k) { return; }
      let row = idx / k;
      let v = clamp(floor(H[idx] * inv[row] + 0.5), -128.0, 127.0);
      Q[idx] = i32(v);`}
    }`;

  // NOTE: the un-batched DP4A matmul that used to live here was removed. It was
  // the only kernel with an exact gate, but the transformer stopped calling it
  // when the block-scaled path landed — so it sat here passing its own gate
  // while verifying nothing that ran. A gate on a kernel nobody calls is worse
  // than no gate: it reads like coverage. The batched kernels below are the
  // ones training uses, and they now carry that exact gate instead.

  // Batched block-scaled GEMM with a FUSED EPILOGUE (CUTLASS ex. 05/24 + 12):
  // grid z = batch index, so ALL attention heads run in ONE dispatch, and the
  // epilogue (block dequant rs·cs + optional ReLU) happens before the data
  // leaves the GPU — f32 out, no int32 readback, no second pass in JS.
  // Each kernel is emitted in two variants from ONE source string: the live one
  // (fused epilogue, f32 out) and a `verify` one that writes the raw int32
  // accumulator instead. The indexing — the part that actually goes wrong — is
  // textually identical, so gating the verify variant genuinely gates the live
  // kernel, and the comparison is EXACT (int8xint8->int32 has no rounding to
  // hide in) instead of an allclose that whole bug classes walk through.
  const OUT_DECL = (v, b) => `@group(0) @binding(${b}) var<storage, read_write> O : array<${v ? "i32" : "f32"}>;`;

  const WGSL_BG_LUT = (verify) => `
    @group(0) @binding(0) var<storage, read> Xq  : array<i32>;   // int8 byte per elem
    @group(0) @binding(1) var<storage, read> Wq  : array<i32>;
    @group(0) @binding(2) var<storage, read> lut : array<i32>;
    @group(0) @binding(3) var<storage, read> rs  : array<f32>;   // per (batch,row)
    @group(0) @binding(4) var<storage, read> cs  : array<f32>;   // per (batch,col)
    ${OUT_DECL(verify, 5)}
    @group(0) @binding(6) var<uniform> dims : vec4<u32>;         // m, k, n, flags(1=relu)
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let k = dims.y; let n = dims.z;
      let row = gid.x; let col = gid.y; let bz = gid.z;
      if (row >= m || col >= n) { return; }
      var s : i32 = 0;
      let xo = (bz * m + row) * k;
      let wo = bz * k * n + col;
      for (var p = 0u; p < k; p = p + 1u) {
        let au = u32(Xq[xo + p] & 255);
        let bu = u32(Wq[wo + p * n] & 255);
        s = s + lut[au * 256u + bu];
      }
      ${verify ? `O[(bz * m + row) * n + col] = s;` : `
      var v = f32(s) * rs[bz * m + row] * cs[bz * n + col];
      if ((dims.w & 1u) == 1u && v < 0.0) { v = 0.0; }
      O[(bz * m + row) * n + col] = v;`}
    }`;

  // same fused/batched kernel on the DP4A hardware path (Wᵀ packed per batch)
  const WGSL_BG_DP4 = (verify) => `
    @group(0) @binding(0) var<storage, read> Xp : array<u32>;
    @group(0) @binding(1) var<storage, read> Wp : array<u32>;    // per-batch Wᵀ, packed
    @group(0) @binding(2) var<storage, read> rs : array<f32>;
    @group(0) @binding(3) var<storage, read> cs : array<f32>;
    ${OUT_DECL(verify, 4)}
    @group(0) @binding(5) var<uniform> dims : vec4<u32>;         // m, kw, n, flags
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let kw = dims.y; let n = dims.z;
      let row = gid.x; let col = gid.y; let bz = gid.z;
      if (row >= m || col >= n) { return; }
      var s : i32 = 0;
      let xo = (bz * m + row) * kw;
      let wo = (bz * n + col) * kw;
      for (var p = 0u; p < kw; p = p + 1u) {
        s = s + dot4I8Packed(Xp[xo + p], Wp[wo + p]);
      }
      ${verify ? `O[(bz * m + row) * n + col] = s;` : `
      var v = f32(s) * rs[bz * m + row] * cs[bz * n + col];
      if ((dims.w & 1u) == 1u && v < 0.0) { v = 0.0; }
      O[(bz * m + row) * n + col] = v;`}
    }`;

  // Gather-fused attention (CUTLASS ex. 36/52): kernels index q/k/v directly in
  // their BT×C layout (head-strided) — no gather copies, no kᵀ transpose — and
  // the ctx kernel scatters straight back into BT×C. int8×int8→i32 is exact, so
  // these are bit-identical to the LUT mirrors (proved at init before use).
  const WGSL_ATT_SCORES = (verify) => `
    @group(0) @binding(0) var<storage, read> Q : array<i32>;     // int8 per elem, BT×C
    @group(0) @binding(1) var<storage, read> K : array<i32>;
    @group(0) @binding(2) var<storage, read> qs : array<f32>;    // per (token,head)
    @group(0) @binding(3) var<storage, read> ks : array<f32>;
    ${OUT_DECL(verify, 4)}
    @group(0) @binding(5) var<uniform> dims : vec4<u32>;         // T, heads, hd, _
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let T = dims.x; let heads = dims.y; let hd = dims.z;
      let ti = gid.x; let tj = gid.y; let bz = gid.z;
      if (ti >= T || tj >= T) { return; }
      let bi = bz / heads; let h = bz % heads;
      let C = heads * hd;
      let qo = (bi * T + ti) * C + h * hd;
      let ko = (bi * T + tj) * C + h * hd;
      var s : i32 = 0;
      for (var p = 0u; p < hd; p = p + 1u) { s = s + Q[qo + p] * K[ko + p]; }
      ${verify ? `O[(bz * T + ti) * T + tj] = s;`
               : `O[(bz * T + ti) * T + tj] = f32(s) * qs[(bi * T + ti) * heads + h] * ks[(bi * T + tj) * heads + h];`}
    }`;
  const WGSL_ATT_CTX = (verify) => `
    @group(0) @binding(0) var<storage, read> A : array<i32>;     // int8, BH×T×T
    @group(0) @binding(1) var<storage, read> V : array<i32>;     // int8, BT×C
    @group(0) @binding(2) var<storage, read> as_ : array<f32>;   // per (bz,row)
    @group(0) @binding(3) var<storage, read> vs : array<f32>;    // per (batch,head,chan)
    ${OUT_DECL(verify, 4)}                                       // BT×C (scatter fused)
    @group(0) @binding(5) var<uniform> dims : vec4<u32>;         // T, heads, hd, _
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let T = dims.x; let heads = dims.y; let hd = dims.z;
      let ti = gid.x; let j = gid.y; let bz = gid.z;
      if (ti >= T || j >= hd) { return; }
      let bi = bz / heads; let h = bz % heads;
      let C = heads * hd;
      let ao = (bz * T + ti) * T;
      var s : i32 = 0;
      for (var tj = 0u; tj < T; tj = tj + 1u) { s = s + A[ao + tj] * V[(bi * T + tj) * C + h * hd + j]; }
      ${verify ? `O[(bi * T + ti) * C + h * hd + j] = s;`
               : `O[(bi * T + ti) * C + h * hd + j] = f32(s) * as_[bz * T + ti] * vs[(bi * heads + h) * hd + j];`}
    }`;

  // Split-K f32 GEMM (CUTLASS ex. 06) for the STE BACKWARD only — the backward
  // was always float (the integer path has no gradient); this just moves that
  // exact float math off the JS thread. Split-K matters for dlnf: M=256, N=32,
  // K=16512 — 8k outputs with a huge inner loop would idle the GPU, so slices
  // of K run on separate workgroups and a second tiny pass reduces partials.
  const WGSL_FGEMM = `
    @group(0) @binding(0) var<storage, read> A : array<f32>;
    @group(0) @binding(1) var<storage, read> Bm : array<f32>;
    @group(0) @binding(2) var<storage, read_write> P : array<f32>;   // S partials
    @group(0) @binding(3) var<uniform> dims : vec4<u32>;             // m, k, n, flags(bit0=transA, rest=S)
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let m = dims.x; let k = dims.y; let n = dims.z;
      let transA = (dims.w & 1u) == 1u;
      let S = dims.w >> 1u;
      let row = gid.x; let col = gid.y; let z = gid.z;
      if (row >= m || col >= n) { return; }
      let ks = (k + S - 1u) / S;
      let p0 = z * ks;
      let p1 = min(k, p0 + ks);
      var s : f32 = 0.0;
      for (var p = p0; p < p1; p = p + 1u) {
        let a = select(A[row * k + p], A[p * m + row], transA);
        s = s + a * Bm[p * n + col];
      }
      P[(z * m + row) * n + col] = s;
    }`;
  const WGSL_FREDUCE = `
    @group(0) @binding(0) var<storage, read> P : array<f32>;
    @group(0) @binding(1) var<storage, read_write> O : array<f32>;
    @group(0) @binding(2) var<uniform> dims : vec4<u32>;             // mn, S, _, _
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let mn = dims.x; let S = dims.y;
      let i = gid.x;
      if (i >= mn) { return; }
      var s : f32 = 0.0;
      for (var z = 0u; z < S; z = z + 1u) { s = s + P[z * mn + i]; }
      O[i] = s;
    }`;

  async function loadLUTs(base) {
    base = base || "";
    const [mulB, reqB, reluB, meta] = await Promise.all([
      fetch(base + "mul_lut.bin").then(r => r.arrayBuffer()),
      fetch(base + "requant_lut.bin").then(r => r.arrayBuffer()),
      fetch(base + "relu_lut.bin").then(r => r.arrayBuffer()),
      fetch(base + "luts_meta.json").then(r => r.json()),
    ]);
    return { mul: new Int16Array(mulB), requant: new Int8Array(reqB),
             relu: new Int8Array(reluB), shift: meta.shift };
  }

  // pack a row-major int8 matrix (rows×cols) into u32 words of 4 bytes along
  // cols, zero-padded to kw words per row (zeros contribute 0 to the dot)
  function packRows(Q, rows, cols, kw) {
    const out = new Uint32Array(rows * kw);
    const bytes = new Uint8Array(out.buffer);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) bytes[(r * kw * 4) + c] = Q[r * cols + c] & 0xFF;
    return out;
  }
  function transposeI8(Q, rows, cols) {
    const out = new Int8Array(rows * cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = Q[r * cols + c];
    return out;
  }

  // f32 gate equality AT THE BIT LEVEL. JS `!==` says -0 === 0, but the fleet
  // compares devices by hashing raw bit patterns (FNV over the byte buffer), so
  // a kernel that flushes -0 to +0 — real hardware has instructions that do
  // exactly this (RDNA2 output modifiers and DX9-legacy multiplies are
  // documented as "not IEEE compatible: -0 is flushed to +0") — would pass a
  // `!==` gate and still fork the weights at the sync guard. The gates must
  // compare the same thing the hash sees: the bits.
  const _fb = new Float32Array(1), _ub = new Uint32Array(_fb.buffer);
  function bitDiff(a, b) { _fb[0] = a; const u = _ub[0]; _fb[0] = b; return u !== _ub[0]; }

  async function initCompute(L) {
    const cpu = { backend: "cpu", label: "CPU (JS)",
                  matmulInt8: (Xq, Wq, m, k, n, LL) => root.Verified.lutMatmulJS(Xq, Wq, m, k, n, LL) };
    if (!(root.navigator && navigator.gpu)) return cpu;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return cpu;
      const device = await adapter.requestDevice();
      const info = adapter.info || {};
      const gpuName = info.description || info.vendor || "WebGPU";

      // LUT pipeline (always built — the fallback and the verification oracle)
      const lutModule = device.createShaderModule({ code: WGSL_LUT });
      const lutPipe = device.createComputePipeline({ layout: "auto", compute: { module: lutModule, entryPoint: "main" } });
      const lut32 = new Int32Array(L.mul);   // widen int16 -> int32
      const lutBuf = device.createBuffer({ size: lut32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(lutBuf, 0, lut32);
      const mkPipe = (code) => device.createComputePipeline({ layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
      // The verify variant doesn't reference the scale buffers, so `layout:auto`
      // would strip those bindings and the bind group would silently mismatch.
      // An EXPLICIT layout keeps both variants binding-compatible — which is the
      // point: they must differ only in the final write, nothing else.
      const mkLayout = (spec) => device.createBindGroupLayout({
        entries: spec.map((t, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: t === "u" ? "uniform" : t === "rw" ? "storage" : "read-only-storage" } })) });
      const mkPipeL = (code, layout) => device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
      const bgLutLayout = mkLayout(["r", "r", "r", "r", "r", "rw", "u"]);
      const bgDp4Layout = mkLayout(["r", "r", "r", "r", "rw", "u"]);
      const attLayout = mkLayout(["r", "r", "r", "r", "rw", "u"]);
      const rowmaxLayout = mkLayout(["r", "rw", "u"]);
      const quantLayout = mkLayout(["r", "r", "rw", "u"]);
      const rowmaxPipe = mkPipeL(WGSL_ROWMAX, rowmaxLayout);
      const quantI32Pipe = mkPipeL(WGSL_QUANT(false), quantLayout);
      const quantPackPipe = mkPipeL(WGSL_QUANT(true), quantLayout);
      // live + verify variants, compiled from the same source (see WGSL_* above)
      const bgLutPipe = mkPipeL(WGSL_BG_LUT(false), bgLutLayout), bgLutVPipe = mkPipeL(WGSL_BG_LUT(true), bgLutLayout);
      const scoresPipe = mkPipeL(WGSL_ATT_SCORES(false), attLayout), scoresVPipe = mkPipeL(WGSL_ATT_SCORES(true), attLayout);
      const ctxPipe = mkPipeL(WGSL_ATT_CTX(false), attLayout), ctxVPipe = mkPipeL(WGSL_ATT_CTX(true), attLayout);

      // gather-fused attention kernels. The gate runs the VERIFY variant of the
      // same source and compares the int32 accumulator with `!==` — exact, no
      // tolerance — then checks the fused epilogue against a bit-exact JS mirror
      // of the WGSL rounding. Swept over several shapes, incl. odd/ragged ones,
      // because head-strided addressing is where these kernels can go wrong.
      let att = { scores: (qq, kq, qs, ks, d) => gpuAttScores(device, d.acc ? scoresVPipe : scoresPipe, qq, kq, qs, ks, d),
                  ctx: (aq, vq, as, vs, d) => gpuAttCtx(device, d.acc ? ctxVPipe : ctxPipe, aq, vq, as, vs, d) };
      try {
        for (const d0 of [{ B: 2, T: 8, heads: 2, hd: 8 }, { B: 1, T: 32, heads: 2, hd: 16 },
                          { B: 3, T: 7, heads: 3, hd: 5 }, { B: 2, T: 33, heads: 4, hd: 8 }]) {
          const nQ = d0.B * d0.T * d0.heads * d0.hd;
          const qq = new Int8Array(nQ), kq = new Int8Array(nQ), vq = new Int8Array(nQ);
          for (let i = 0; i < nQ; i++) { qq[i] = (Math.random() * 256 - 128) | 0; kq[i] = (Math.random() * 256 - 128) | 0; vq[i] = (Math.random() * 256 - 128) | 0; }
          const qs = Float32Array.from({ length: d0.B * d0.T * d0.heads }, () => Math.random() + 0.5);
          const ks = Float32Array.from({ length: d0.B * d0.T * d0.heads }, () => Math.random() + 0.5);
          const aq = new Int8Array(d0.B * d0.heads * d0.T * d0.T);
          for (let i = 0; i < aq.length; i++) aq[i] = (Math.random() * 127) | 0;
          const as = Float32Array.from({ length: d0.B * d0.heads * d0.T }, () => Math.random() + 0.5);
          const vs = Float32Array.from({ length: d0.B * d0.heads * d0.hd }, () => Math.random() + 0.5);
          const dv = { ...d0, acc: true };
          const [accS, accC, hwS, hwC] = await Promise.all([
            att.scores(qq, kq, qs, ks, dv), att.ctx(aq, vq, as, vs, dv),
            att.scores(qq, kq, qs, ks, d0), att.ctx(aq, vq, as, vs, d0)]);
          const refAccS = root.Verified.attScoresJS(qq, kq, qs, ks, dv, L);
          const refAccC = root.Verified.attCtxJS(aq, vq, as, vs, dv, L);
          for (let i = 0; i < refAccS.length; i++) if (accS[i] !== refAccS[i]) throw new Error(`scores accumulator mismatch @${i} shape ${JSON.stringify(d0)}`);
          for (let i = 0; i < refAccC.length; i++) if (accC[i] !== refAccC[i]) throw new Error(`ctx accumulator mismatch @${i} shape ${JSON.stringify(d0)}`);
          const refS = root.Verified.attScoresJS(qq, kq, qs, ks, d0, L);
          const refC = root.Verified.attCtxJS(aq, vq, as, vs, d0, L);
          for (let i = 0; i < refS.length; i++) if (bitDiff(hwS[i], refS[i])) throw new Error(`scores epilogue mismatch @${i}`);
          for (let i = 0; i < refC.length; i++) if (bitDiff(hwC[i], refC[i])) throw new Error(`ctx epilogue mismatch @${i}`);
        }
      } catch (e) {
        console.warn("fused attention kernels failed verification — using CPU LUT mirrors:", e.message);
        att = null;
      }

      // split-K f32 GEMM for the STE backward (self-tested vs JS float matmul)
      const fPipes = { gemm: mkPipe(WGSL_FGEMM), reduce: mkPipe(WGSL_FREDUCE) };
      let fgemm = (A, Bm, d) => gpuFgemm(device, fPipes, A, Bm, d);
      let fgemm2 = (A, B1, d1, B2, d2) => gpuFgemm2(device, fPipes, A, B1, d1, B2, d2);
      let fgemmFma = false;                     // set by the gate below
      try {
        const m0 = 7, k0 = 4500, n0 = 5;                      // k big enough to exercise split-K
        const A = Float32Array.from({ length: m0 * k0 }, () => Math.random() - 0.5);
        const Bm = Float32Array.from({ length: k0 * n0 }, () => Math.random() - 0.5);
        // EXACT gate, replacing an allclose. The mirror reproduces split-K's
        // partition and accumulation order, so bit-equality is achievable and
        // is the honest bar. WGSL may contract `s + a*b` into an FMA, so try
        // both rounding schedules and record which the device implements —
        // if NEITHER matches, the f32 backward is not bit-reproducible here
        // and must not run, because its gradients set the weights the whole
        // fleet is hashed against.
        const dG = { m: m0, k: k0, n: n0 };
        const hw = await fgemm(A, Bm, dG);
        const mStep = root.Verified.fgemmMirror(A, Bm, dG, false);
        const mFma = root.Verified.fgemmMirror(A, Bm, dG, true);
        let okStep = true, okFma = true;
        for (let i = 0; i < hw.length; i++) {
          if (bitDiff(hw[i], mStep[i])) okStep = false;
          if (bitDiff(hw[i], mFma[i])) okFma = false;
          if (!okStep && !okFma) break;
        }
        if (!okStep && !okFma) throw new Error("fgemm matches neither rounding schedule — not bit-reproducible");
        fgemmFma = !okStep;                     // remember what this device does
        console.info(`split-K f32 GEMM is bit-exact (${okStep ? "two-rounding" : "fused"} multiply-add schedule)`);
        // The shared-operand pair must equal the two calls it replaces, EXACTLY:
        // same buffer, same shader, same arithmetic, so bit-level equality is
        // the honest bar (not a tolerance). Both index patterns of A are
        // exercised — transA on one leg, plain on the other, the live shapes.
        const kT = 300, mT = 9, nT = 6;                       // A is mT x kT, used both ways
        const A2 = Float32Array.from({ length: mT * kT }, () => Math.random() - 0.5);
        const Bt = Float32Array.from({ length: mT * nT }, () => Math.random() - 0.5);
        const Bn = Float32Array.from({ length: kT * nT }, () => Math.random() - 0.5);
        const dT = { m: kT, k: mT, n: nT, transA: true }, dN = { m: mT, k: kT, n: nT };
        const [f1, f2] = await fgemm2(A2, Bt, dT, Bn, dN);
        const [r1, r2] = [await fgemm(A2, Bt, dT), await fgemm(A2, Bn, dN)];
        for (let i = 0; i < r1.length; i++) if (bitDiff(f1[i], r1[i])) throw new Error("fgemm2 transA leg differs");
        for (let i = 0; i < r2.length; i++) if (bitDiff(f2[i], r2[i])) throw new Error("fgemm2 plain leg differs");
      } catch (e) {
        console.warn("split-K f32 GEMM failed verification — backward stays in JS:", e.message);
        fgemm = null; fgemm2 = null;
      }

      // Shared exact gate for a bgemm implementation. Sweeps shapes (including
      // ragged ones and a k long enough to matter), compares the int32
      // accumulator from the verify variant with `!==`, then compares the fused
      // f32 epilogue against the bit-exact JS mirror. Returns null if clean.
      async function gateBgemm(bgFn) {
        for (const d0 of [{ m: 5, k: 9, n: 6, batch: 3, relu: true },
                          { m: 32, k: 64, n: 32, batch: 1, relu: false },
                          { m: 7, k: 253, n: 5, batch: 2, relu: true },
                          { m: 1, k: 4, n: 1, batch: 1, relu: false },
                          { m: 17, k: 33, n: 9, batch: 1, relu: true }]) {
          const Xq = new Int8Array(d0.batch * d0.m * d0.k), Wq = new Int8Array(d0.batch * d0.k * d0.n);
          for (let i = 0; i < Xq.length; i++) Xq[i] = (Math.random() * 256 - 128) | 0;
          for (let i = 0; i < Wq.length; i++) Wq[i] = (Math.random() * 256 - 128) | 0;
          const rs = Float32Array.from({ length: d0.batch * d0.m }, () => Math.random() + 0.5);
          const cs = Float32Array.from({ length: d0.batch * d0.n }, () => Math.random() + 0.5);
          const shape = `${d0.m}x${d0.k}x${d0.n}b${d0.batch}`;
          const accHw = await bgFn(Xq, Wq, rs, cs, { ...d0, acc: true });
          const accRef = root.Verified.bgemmJS(Xq, Wq, rs, cs, { ...d0, acc: true }, L);
          for (let i = 0; i < accRef.length; i++)
            if (accHw[i] !== accRef[i]) return `accumulator mismatch @${i} (${shape}): ${accHw[i]} vs ${accRef[i]}`;
          const hw = await bgFn(Xq, Wq, rs, cs, d0);
          const ref = root.Verified.bgemmJS(Xq, Wq, rs, cs, d0, L);
          for (let i = 0; i < ref.length; i++)
            if (bitDiff(hw[i], ref[i])) return `epilogue mismatch @${i} (${shape}): ${hw[i]} vs ${ref[i]}`;
        }
        return null;
      }

      // Poison the pool with large-magnitude residue at the gate's own shapes,
      // so a re-gate runs on RECYCLED (non-zero) buffers. See the dirty-buffer
      // gate note below the MLP gate for why this matters.
      async function poisonBgemm(bgFn) {
        for (const d0 of [{ m: 5, k: 9, n: 6, batch: 3, relu: true },
                          { m: 32, k: 64, n: 32, batch: 1, relu: false },
                          { m: 7, k: 253, n: 5, batch: 2, relu: true },
                          { m: 1, k: 4, n: 1, batch: 1, relu: false },
                          { m: 17, k: 33, n: 9, batch: 1, relu: true }]) {
          const Xq = new Int8Array(d0.batch * d0.m * d0.k).fill(127);
          const Wq = new Int8Array(d0.batch * d0.k * d0.n).fill(127);
          const rs = new Float32Array(d0.batch * d0.m).fill(1e4);
          const cs = new Float32Array(d0.batch * d0.n).fill(1e4);
          await bgFn(Xq, Wq, rs, cs, d0);
          await bgFn(Xq, Wq, rs, cs, { ...d0, acc: true });
        }
      }
      const gateBgemmDirty = async (bgFn) => { await poisonBgemm(bgFn); return gateBgemm(bgFn); };

      const bgLut = (Xq, Wq, rs, cs, d) => gpuBgemmLUT(device, d.acc ? bgLutVPipe : bgLutPipe, lutBuf, Xq, Wq, rs, cs, d);
      // the LUT bgemm is the fallback AND the oracle's shader twin — gate it too
      const lutBad = (await gateBgemm(bgLut)) || (await gateBgemmDirty(bgLut));
      if (lutBad) { console.warn("LUT bgemm shader failed verification — CPU mirrors only:", lutBad); return cpu; }

      // B2B MLP chain gate (CUTLASS ex. 13+23): run the WHOLE chain — gemm1 +
      // ReLU + on-GPU rowmax + on-GPU quantize + gemm2 — against the pure-JS
      // mirror chain, exact `!==` on both h1 and the final output. Sweeps
      // ragged shapes; h not a multiple of 4 exercises the pack-tail padding.
      const MLP_SHAPES = [{ m: 6, k: 8, h: 12, n: 5 }, { m: 5, k: 16, h: 6, n: 3 },
                          { m: 17, k: 33, h: 10, n: 9 }, { m: 32, k: 64, h: 128, n: 32 }];
      async function gateMlp(mlpFn, sweepOnly) {
        for (const d0 of MLP_SHAPES) {
          const rnd = (len) => Float32Array.from({ length: len }, () => Math.random() * 2 - 1);
          const Xf = rnd(d0.m * d0.k), W1 = rnd(d0.k * d0.h), W2 = rnd(d0.h * d0.n);
          const hw = await root.Verified.vmlpBlock(Xf, W1, W2, d0, L, mlpFn, null);
          const ref = await root.Verified.vmlpBlock(Xf, W1, W2, d0, L, null, null);
          const shape = `${d0.m}x${d0.k}x${d0.h}x${d0.n}`;
          for (let i = 0; i < ref.h1.length; i++)
            if (bitDiff(hw.h1[i], ref.h1[i])) return `h1 mismatch @${i} (${shape}): ${hw.h1[i]} vs ${ref.h1[i]}`;
          for (let i = 0; i < ref.out.length; i++)
            if (bitDiff(hw.out[i], ref.out[i])) return `out mismatch @${i} (${shape}): ${hw.out[i]} vs ${ref.out[i]}`;
        }
        if (sweepOnly) return null;      // dirty re-gate: sweep is the point, skip the respec hunt
        // DISCRIMINATING case: the sweep above passes vacuously if no value
        // lands on a rounding boundary — the old round(x/scale) spec would
        // pass it too. So hunt (in fast JS) for an input where the two specs
        // actually disagree, then check the GPU sides with the RESPEC. This
        // gates the gate: a pass must be something the old spec would fail.
        const V = root.Verified;
        const d1 = { m: 16, k: 32, h: 64, n: 16 };
        const rnd1 = (len) => Float32Array.from({ length: len }, () => Math.random() * 4 - 2);
        for (let t = 0; t < 800; t++) {
          const Xf = rnd1(d1.m * d1.k), W1 = rnd1(d1.k * d1.h), W2 = rnd1(d1.h * d1.n);
          const x = V.quantizeRows(Xf, d1.m, d1.k), w1 = V.quantizeCols(W1, d1.k, d1.h);
          const h1 = V.bgemmJS(x.q, w1.q, x.s, w1.s, { m: d1.m, k: d1.k, n: d1.h, batch: 1, relu: true }, L);
          const sc = V.scalesFromAbsMax(V.rowAbsMax(h1, d1.m, d1.h));
          const qNew = V.quantizeRowsInv(h1, d1.m, d1.h, sc.inv);
          const qOld = V.quantizeRows(h1, d1.m, d1.h).q;
          let boundary = false;
          for (let i = 0; i < qNew.length; i++) if (qNew[i] !== qOld[i]) { boundary = true; break; }
          if (!boundary) continue;
          const w2 = V.quantizeCols(W2, d1.h, d1.n);
          const gpu = await mlpFn(x.q, w1.q, w2.q, x.s, w1.s, w2.s, d1);
          const refNew = V.bgemmJS(qNew, w2.q, sc.scale, w2.s, { m: d1.m, k: d1.h, n: d1.n, batch: 1 }, L);
          const refOld = V.bgemmJS(qOld, w2.q, sc.scale, w2.s, { m: d1.m, k: d1.h, n: d1.n, batch: 1 }, L);
          let eqNew = true, eqOld = true;
          for (let i = 0; i < refNew.length; i++) { if (bitDiff(gpu.out[i], refNew[i])) eqNew = false; if (bitDiff(gpu.out[i], refOld[i])) eqOld = false; }
          if (!eqNew) return "discriminating boundary case: GPU chain does not match the respec mirror";
          if (eqOld) return "discriminating boundary case: GPU chain matches the OLD quantize spec";
          return null;                                    // proven: respec, not merely gate-compatible
        }
        console.warn("B2B gate: no rounding-boundary input found in 800 trials — respec discrimination unproven this boot (sweep still exact)");
        return null;
      }
      // FMA-contraction note for the quantize kernel: WGSL permits a compiler
      // to contract `H*inv + 0.5` into a hardware FMA (e.g. RDNA2 V_FMA_F32 —
      // ONE rounding instead of two). This CANNOT change the quantized int8:
      // adding 0.5 is exact except at binade crossings, and there the
      // double-rounding anomaly only moves the value within the same integer
      // cell (the RNE tie parity resolves both schedules to the same side of
      // the integer), so floor() sees no difference. Verified empirically in
      // test_b2b.js: 48M draws targeted at binade edges, 1.9M last-ulp
      // fused-vs-stepped differences, ZERO floor-visible. The `+0.5` respec is
      // contraction-immune by construction — `round(x/scale)` was not.
      // ---- DIRTY-BUFFER GATE ----------------------------------------------
      // A pooled buffer is NOT zero-initialized, so any kernel that assumes
      // zeros (the rowmax atomicMax accumulator does) is right on step one and
      // wrong on step two. That is a STATE bug: no single call is wrong, the
      // sequence is — the family an oracle cannot reach.
      //
      // MEASURED, and it corrected the assumption that motivated this code:
      // the existing sweep ALREADY catches it. Deleting the clearBuffer and
      // re-running showed the plain gate failing at the SECOND shape, because
      // the sweep's own shapes recycle each other's buffers (same power-of-2
      // bucket) and an uncleared max only grows. So the suite was never
      // blind — it had incidental dirty coverage nobody designed.
      //
      // Incidental is the problem. It relies on the sweep having >= 2 shapes,
      // on them colliding in one bucket, and on the residue exceeding the real
      // value. Change the shape list and the coverage silently evaporates.
      // The poison below makes it deliberate: run first at 1e4 magnitude so
      // the residue dominates ANY value the gate can produce, release it, then
      // sweep again. Detection stops depending on ordering luck and covers the
      // first shape too. Cheap by design — the re-gate skips the 800-trial
      // respec hunt, which has nothing to do with buffer state.
      async function poisonPool(mlpFn) {
        // same shapes as the gate => same pool buckets => the gate's next
        // acquisition is exactly one of these poisoned buffers (free list is LIFO)
        const big = (len) => Float32Array.from({ length: len }, () => (Math.random() * 2 - 1) * 1e4);
        for (const d0 of MLP_SHAPES)
          await root.Verified.vmlpBlock(big(d0.m * d0.k), big(d0.k * d0.h), big(d0.h * d0.n), d0, L, mlpFn, null);
      }
      async function gateMlpDirty(mlpFn) {
        await poisonPool(mlpFn);
        return gateMlp(mlpFn, true);    // sweep only, on recycled non-zero buffers
      }

      const lutMlpEnv = { dp4: false, gemm: bgLutPipe, rowmax: rowmaxPipe, quant: quantI32Pipe, lutBuf };
      let mlpLut = (xq, w1q, w2q, xs, w1s, w2s, d) => gpuMlpChain(device, lutMlpEnv, xq, w1q, w2q, xs, w1s, w2s, d);
      const mlpLutBad = (await gateMlp(mlpLut)) || (await gateMlpDirty(mlpLut));
      if (mlpLutBad) { console.warn("B2B MLP chain (LUT) failed verification — MLP stays on the CPU mirror chain:", mlpLutBad); mlpLut = null; }

      const viaLUT = { backend: "webgpu", label: `${gpuName} (LUT shader · exact-gated)`,
                       matmulInt8: (Xq, Wq, m, k, n) => gpuMatmulLUT(device, lutPipe, lutBuf, Xq, Wq, m, k, n),
                       bgemm: bgLut, att, fgemm, fgemm2, mlp: mlpLut };

      // DP4A pipeline — only if the WGSL feature exists AND its batched kernel
      // reproduces the verified units exactly across the shape sweep
      if (!(navigator.gpu.wgslLanguageFeatures && navigator.gpu.wgslLanguageFeatures.has("packed_4x8_integer_dot_product")))
        return viaLUT;
      const bgDp4Pipe = mkPipeL(WGSL_BG_DP4(false), bgDp4Layout), bgDp4VPipe = mkPipeL(WGSL_BG_DP4(true), bgDp4Layout);
      const bg = (Xq, Wq, rs, cs, d) => gpuBgemmDP4(device, d.acc ? bgDp4VPipe : bgDp4Pipe, Xq, Wq, rs, cs, d);
      const dp4Bad = (await gateBgemm(bg)) || (await gateBgemmDirty(bg));
      if (dp4Bad) {
        console.warn("batched DP4A disagreed with the verified units — using LUT bgemm:", dp4Bad);
        return viaLUT;
      }
      const dp4MlpEnv = { dp4: true, gemm: bgDp4Pipe, rowmax: rowmaxPipe, quant: quantPackPipe };
      let mlpDp4 = (xq, w1q, w2q, xs, w1s, w2s, d) => gpuMlpChain(device, dp4MlpEnv, xq, w1q, w2q, xs, w1s, w2s, d);
      const mlpDp4Bad = (await gateMlp(mlpDp4)) || (await gateMlpDirty(mlpDp4));
      if (mlpDp4Bad) { console.warn("B2B MLP chain (DP4A) failed verification — using the LUT chain:", mlpDp4Bad); mlpDp4 = mlpLut; }

      // Both backends are exact-gated bit-identical, so which one runs is a
      // free choice — and an init race that timed them was tried and REMOVED.
      // Measured on this NVIDIA part they tie on the shipped float path (DP4A's
      // JS packing cancels its dot-throughput win at these sizes), so the race
      // bought nothing, cost ~40 ms of init, and made the backend vary between
      // page loads — which silently invalidated three separate A/B benchmarks
      // before it was noticed. A knob that changes what you are measuring is
      // worse than a fixed choice. DP4A ships: it wins the int8-backward mode
      // by ~12% and ties elsewhere.
      return { backend: "webgpu", label: `${gpuName} (DP4A int8 dot · exact-gated vs units)`,
               bgemm: bg, att, fgemm, fgemm2, mlp: mlpDp4 };
    } catch (e) { console.warn("WebGPU init failed, CPU fallback:", e); return cpu; }
  }

  // shared dispatch/readback plumbing
  async function runPass(device, pipeline, entries, m, n) {
    const bytesC = m * n * 4;
    const bufC = mk(device, bytesC, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: entries(bufC) });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8)); pass.end();
    const read = mk(device, bytesC, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(bufC, 0, read, 0, bytesC);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Int32Array(read.getMappedRange(0, bytesC).slice(0));   // pooled: map only the logical range
    read.unmap();
    return { out, bufC, read };
  }

  async function gpuMatmulLUT(device, pipeline, lutBuf, Xq, Wq, m, k, n) {
    const X32 = Int32Array.from(Xq), W32 = Int32Array.from(Wq);   // byte -> i32
    const bufX = up(device, X32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = up(device, W32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([m, k, n, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runPass(device, pipeline, (bufC) => [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: lutBuf } }, { binding: 3, resource: { buffer: bufC } },
      { binding: 4, resource: { buffer: bufD } } ], m, n);
    release(device, [bufX, bufW, bufD, r.bufC, r.read]);
    return r.out;
  }

  // attention kernels: int8 (widened i32) in, f32 out, strided head indexing
  async function gpuAttScores(device, pipeline, qq, kq, qs, ks, d) {
    const { B, T, heads } = d;
    const bufQ = up(device, Int32Array.from(qq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufK = up(device, Int32Array.from(kq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufQs = up(device, qs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufKs = up(device, ks, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([d.T, d.heads, d.hd, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufQ } }, { binding: 1, resource: { buffer: bufK } },
      { binding: 2, resource: { buffer: bufQs } }, { binding: 3, resource: { buffer: bufKs } },
      { binding: 4, resource: { buffer: bufO } }, { binding: 5, resource: { buffer: bufD } } ], T, T, B * heads, d.acc);
    release(device, [bufQ, bufK, bufQs, bufKs, bufD, r.bufO, r.read]);
    return r.out;
  }
  async function gpuAttCtx(device, pipeline, aq, vq, as, vs, d) {
    const { B, T, heads, hd } = d;
    const bufA = up(device, Int32Array.from(aq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufV = up(device, Int32Array.from(vq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufAs = up(device, as, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufVs = up(device, vs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([T, heads, hd, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufV } },
      { binding: 2, resource: { buffer: bufAs } }, { binding: 3, resource: { buffer: bufVs } },
      { binding: 4, resource: { buffer: bufO } }, { binding: 5, resource: { buffer: bufD } } ], T, hd, B * heads, d.acc);
    release(device, [bufA, bufV, bufAs, bufVs, bufD, r.bufO, r.read]);
    return r.out;
  }

  // split-K f32 GEMM (backward): partial pass + reduce pass
  async function gpuFgemm(device, pipes, A, Bm, d) {
    const { m, k, n } = d, transA = d.transA ? 1 : 0;
    const S = k > 4096 ? Math.min(16, Math.ceil(k / 2048)) : 1;
    const bufA = up(device, A, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufB = up(device, Bm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufP = mk(device, S * m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bufD1 = up(device, new Uint32Array([m, k, n, transA | (S << 1)]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const bufO = mk(device, m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bufD2 = up(device, new Uint32Array([m * n, S, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(pipes.gemm);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipes.gemm.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufP } }, { binding: 3, resource: { buffer: bufD1 } } ] }));
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8), S); pass.end();
    pass = enc.beginComputePass();
    pass.setPipeline(pipes.reduce);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipes.reduce.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: bufP } }, { binding: 1, resource: { buffer: bufO } },
      { binding: 2, resource: { buffer: bufD2 } } ] }));
    pass.dispatchWorkgroups(Math.ceil(m * n / 64)); pass.end();
    const read = mk(device, m * n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(bufO, 0, read, 0, m * n * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(read.getMappedRange(0, m * n * 4).slice(0));   // pooled: logical range only
    read.unmap();
    release(device, [bufA, bufB, bufP, bufD1, bufO, bufD2, read]);
    return out;
  }

  // SHARED-OPERAND fused f32 GEMM pair. The two embedding-gradient GEMMs both
  // consume dlogits (BT x vocab). With the 16512-token vocab that operand is
  // ~17 MB, and running them as two independent calls uploaded it TWICE and
  // paid two submits and two map round trips — profiling put the pair at 55%
  // of the whole step. Here A goes up ONCE, both GEMM+reduce chains ride one
  // command encoder, one submit covers both, and the two readbacks are mapped
  // concurrently. The shader reads A as either A[row*k+p] or A[p*m+row]
  // (transA), so ONE flat buffer serves both index patterns — nothing about
  // the arithmetic changes, which is why the gradients stay bit-identical.
  async function gpuFgemm2(device, pipes, A, B1, d1, B2, d2) {
    const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const UU = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const bufA = up(device, A, SU);                       // the shared 17 MB operand: ONE upload
    const enc = device.createCommandEncoder();
    const legs = [];
    for (const [Bm, d] of [[B1, d1], [B2, d2]]) {
      const { m, k, n } = d, transA = d.transA ? 1 : 0;
      const S = k > 4096 ? Math.min(16, Math.ceil(k / 2048)) : 1;
      const bufB = up(device, Bm, SU);
      const bufP = mk(device, S * m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const bufD1 = up(device, new Uint32Array([m, k, n, transA | (S << 1)]), UU);
      const bufO = mk(device, m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const bufD2 = up(device, new Uint32Array([m * n, S, 0, 0]), UU);
      let pass = enc.beginComputePass();
      pass.setPipeline(pipes.gemm);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipes.gemm.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufP } }, { binding: 3, resource: { buffer: bufD1 } } ] }));
      pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8), S); pass.end();
      pass = enc.beginComputePass();
      pass.setPipeline(pipes.reduce);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipes.reduce.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: bufP } }, { binding: 1, resource: { buffer: bufO } },
        { binding: 2, resource: { buffer: bufD2 } } ] }));
      pass.dispatchWorkgroups(Math.ceil(m * n / 64)); pass.end();
      const bytes = m * n * 4;
      const read = mk(device, bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      enc.copyBufferToBuffer(bufO, 0, read, 0, bytes);
      legs.push({ read, bytes, bufs: [bufB, bufP, bufD1, bufO, bufD2] });
    }
    device.queue.submit([enc.finish()]);                  // ONE submit for both GEMMs
    await Promise.all(legs.map((l) => l.read.mapAsync(GPUMapMode.READ)));
    const outs = legs.map((l) => {
      const o = new Float32Array(l.read.getMappedRange(0, l.bytes).slice(0));
      l.read.unmap();
      return o;
    });
    release(device, [bufA, ...legs.flatMap((l) => [...l.bufs, l.read])]);
    return outs;
  }

  // fused batched dispatch: f32 out, epilogue done on-device (raw=true reads the
  // verify variant's int32 accumulator instead)
  async function runBgPass(device, pipeline, entries, m, n, batch, raw) {
    const bytesO = batch * m * n * 4;
    const bufO = mk(device, bytesO, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries(bufO) });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8), batch); pass.end();
    const read = mk(device, bytesO, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(bufO, 0, read, 0, bytesO);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const buf = read.getMappedRange(0, bytesO).slice(0);   // pooled: logical range only
    const out = raw ? new Int32Array(buf) : new Float32Array(buf);
    read.unmap();
    return { out, bufO, read };
  }

  async function gpuBgemmLUT(device, pipeline, lutBuf, Xq, Wq, rs, cs, d) {
    const { m, k, n } = d, batch = d.batch || 1, flags = d.relu ? 1 : 0;
    const bufX = up(device, Int32Array.from(Xq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = up(device, Int32Array.from(Wq), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufR = up(device, rs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufS = up(device, cs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([m, k, n, flags]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: lutBuf } }, { binding: 3, resource: { buffer: bufR } },
      { binding: 4, resource: { buffer: bufS } }, { binding: 5, resource: { buffer: bufO } },
      { binding: 6, resource: { buffer: bufD } } ], m, n, batch, d.acc);
    release(device, [bufX, bufW, bufR, bufS, bufD, r.bufO, r.read]);
    return r.out;
  }

  async function gpuBgemmDP4(device, pipeline, Xq, Wq, rs, cs, d) {
    const { m, k, n } = d, batch = d.batch || 1, flags = d.relu ? 1 : 0;
    const kw = Math.ceil(k / 4);
    const Xp = packRows(Xq, batch * m, k, kw);            // rows are (batch·m)
    const Wp = new Uint32Array(batch * n * kw);           // per-batch Wᵀ, packed
    for (let bz = 0; bz < batch; bz++) {
      const wt = transposeI8(Wq.subarray(bz * k * n, (bz + 1) * k * n), k, n);
      Wp.set(packRows(wt, n, k, kw), bz * n * kw);
    }
    const bufX = up(device, Xp, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufW = up(device, Wp, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufR = up(device, rs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufS = up(device, cs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bufD = up(device, new Uint32Array([m, kw, n, flags]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const r = await runBgPass(device, pipeline, (bufO) => [
      { binding: 0, resource: { buffer: bufX } }, { binding: 1, resource: { buffer: bufW } },
      { binding: 2, resource: { buffer: bufR } }, { binding: 3, resource: { buffer: bufS } },
      { binding: 4, resource: { buffer: bufO } }, { binding: 5, resource: { buffer: bufD } } ], m, n, batch, d.acc);
    release(device, [bufX, bufW, bufR, bufS, bufD, r.bufO, r.read]);
    return r.out;
  }

  // B2B MLP chain: gemm1 (ReLU fused) + rowmax in one encoder, a 4·m-byte
  // absmax readback, then quantize + gemm2 in a second encoder. h1 comes back
  // because the STE backward needs it, but it never goes UP again — gemm2's
  // left operand is produced and consumed entirely on the GPU.
  async function gpuMlpChain(device, env, xq, w1q, w2q, xs, w1s, w2s, d) {
    const { m, k, h, n } = d, dp4 = !!env.dp4;
    const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    let bufX, bufW1, bufW2, kw1, hw;
    if (dp4) {
      kw1 = Math.ceil(k / 4); hw = Math.ceil(h / 4);
      bufX = up(device, packRows(xq, m, k, kw1), SU);
      bufW1 = up(device, packRows(transposeI8(w1q, k, h), h, k, kw1), SU);
      bufW2 = up(device, packRows(transposeI8(w2q, h, n), n, h, hw), SU);
    } else {
      kw1 = k; hw = h;
      bufX = up(device, Int32Array.from(xq), SU);
      bufW1 = up(device, Int32Array.from(w1q), SU);
      bufW2 = up(device, Int32Array.from(w2q), SU);
    }
    const bufRs = up(device, xs, SU), bufCs1 = up(device, w1s, SU), bufCs2 = up(device, w2s, SU);
    const bufH = mk(device, m * h * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    // rowmax accumulates with atomicMax and NEEDS zeros. A fresh buffer was
    // zero-initialized; a POOLED buffer is not — cleared in the encoder below.
    const bufMX = mk(device, m * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const UU = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const bufD1 = up(device, new Uint32Array([m, kw1, h, 1]), UU);       // flags=1: fused ReLU
    const bufDM = up(device, new Uint32Array([m, h, 0, 0]), UU);
    const gemmBind = (bufA, bufB, bufR, bufC, bufO, bufD) => device.createBindGroup({
      layout: env.gemm.getBindGroupLayout(0),
      entries: (dp4 ? [bufA, bufB, bufR, bufC, bufO, bufD]
                    : [bufA, bufB, env.lutBuf, bufR, bufC, bufO, bufD])
        .map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const enc1 = device.createCommandEncoder();
    enc1.clearBuffer(bufMX, 0, m * 4);            // pooled buffer: zero the atomicMax accumulator
    let pass = enc1.beginComputePass();
    pass.setPipeline(env.gemm);
    pass.setBindGroup(0, gemmBind(bufX, bufW1, bufRs, bufCs1, bufH, bufD1));
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(h / 8), 1); pass.end();
    pass = enc1.beginComputePass();
    pass.setPipeline(env.rowmax);
    pass.setBindGroup(0, device.createBindGroup({ layout: env.rowmax.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: bufH } }, { binding: 1, resource: { buffer: bufMX } },
      { binding: 2, resource: { buffer: bufDM } } ] }));
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(h / 8), 1); pass.end();
    const readH = mk(device, m * h * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const readM = mk(device, m * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc1.copyBufferToBuffer(bufH, 0, readH, 0, m * h * 4);
    enc1.copyBufferToBuffer(bufMX, 0, readM, 0, m * 4);
    device.queue.submit([enc1.finish()]);
    await Promise.all([readH.mapAsync(GPUMapMode.READ), readM.mapAsync(GPUMapMode.READ)]);
    const h1 = new Float32Array(readH.getMappedRange(0, m * h * 4).slice(0)); readH.unmap();
    // the atomicMax'ed u32 bit patterns ARE the f32 |max| values
    const mx = new Float32Array(readM.getMappedRange(0, m * 4).slice(0)); readM.unmap();
    // scale derivation in JS f64 — exactly rounded, identical on every device
    // (WGSL division is 2.5 ULP, which is why it never runs on the GPU)
    const sc = root.Verified.scalesFromAbsMax(mx);
    const bufInv = up(device, sc.inv, SU), bufHs = up(device, sc.scale, SU);
    const bufQ = mk(device, m * hw * 4, GPUBufferUsage.STORAGE);
    const bufDQ = up(device, new Uint32Array([m, h, hw, 0]), UU);
    const bufD2 = up(device, new Uint32Array([m, hw, n, 0]), UU);
    const bufO = mk(device, m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const enc2 = device.createCommandEncoder();
    pass = enc2.beginComputePass();
    pass.setPipeline(env.quant);
    pass.setBindGroup(0, device.createBindGroup({ layout: env.quant.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: bufH } }, { binding: 1, resource: { buffer: bufInv } },
      { binding: 2, resource: { buffer: bufQ } }, { binding: 3, resource: { buffer: bufDQ } } ] }));
    pass.dispatchWorkgroups(Math.ceil((m * (dp4 ? hw : h)) / 64)); pass.end();
    pass = enc2.beginComputePass();
    pass.setPipeline(env.gemm);
    pass.setBindGroup(0, gemmBind(bufQ, bufW2, bufHs, bufCs2, bufO, bufD2));
    pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8), 1); pass.end();
    const readO = mk(device, m * n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc2.copyBufferToBuffer(bufO, 0, readO, 0, m * n * 4);
    device.queue.submit([enc2.finish()]);
    await readO.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readO.getMappedRange(0, m * n * 4).slice(0)); readO.unmap();
    release(device, [bufX, bufW1, bufW2, bufRs, bufCs1, bufCs2, bufH, bufMX, bufD1, bufDM,
     bufInv, bufHs, bufQ, bufDQ, bufD2, bufO, readH, readM, readO]);
    return { h1, out };
  }

  // ---- buffer pool ------------------------------------------------------------
  // Every dispatch used to create and destroy its buffers — ~19 create/destroy
  // pairs per MLP chain call, per layer, per step. Creation cost is pure driver
  // overhead, and at these GEMM sizes it is a large share of the step. The pool
  // recycles buffers by (usage, size-bucket): same bytes uploaded, same regions
  // read, ZERO math change — the kernels only ever index inside the logical
  // dims, so the stale tail of a bucketed buffer is never read. The one
  // exception is a buffer a kernel expects ZEROED (the rowmax accumulator):
  // pooled buffers are NOT fresh, so that one is cleared explicitly in the
  // encoder (see gpuMlpChain). Correctness is enforced where it always was:
  // the exact init gates, the live audit, and the per-step probe hash all run
  // through these pooled paths.
  const _pools = new WeakMap();                     // device -> Map(key -> free list)
  function _poolOf(device) {
    let p = _pools.get(device);
    if (!p) { p = new Map(); _pools.set(device, p); }
    return p;
  }
  function mk(device, size, usage) {
    let cap = 256; while (cap < size) cap *= 2;
    const key = usage + ":" + cap;
    const list = _poolOf(device).get(key);
    if (list && list.length) return list.pop();
    const b = device.createBuffer({ size: cap, usage });
    b._poolKey = key;
    return b;
  }
  function up(device, arr, usage) {
    const b = mk(device, Math.max(16, arr.byteLength), usage);
    device.queue.writeBuffer(b, 0, arr);
    return b;
  }
  function release(device, bufs) {
    const p = _poolOf(device);
    for (const b of bufs) {
      if (!b || !b._poolKey) { if (b) b.destroy(); continue; }
      let list = p.get(b._poolKey);
      if (!list) { list = []; p.set(b._poolKey, list); }
      if (list.length < 64) list.push(b); else b.destroy();
    }
  }

  // ---- canonical kernel probe -------------------------------------------------
  // The weight hash CANNOT catch a device whose kernel is wrong: weights only
  // depend on the gradient bytes everyone receives, so a fleet averaging one
  // device's bad gradient stays bit-identical and perfectly happy. This is the
  // check that can. Every device runs the SAME seeded int8 GEMM through its own
  // live kernel (verify variant -> raw int32 accumulator, exact on every
  // backend — GPU DP4A, GPU LUT, CPU mirror alike) and hashes the result. Same
  // input + correct kernels => same hash, regardless of hardware. A device that
  // disagrees is computing different arithmetic than the rest of the fleet.
  const PROBE = { m: 24, k: 96, n: 24, batch: 2 };
  function probeInputs() {
    let seed = 0x5EED;                       // fixed: identical on every device
    const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const { m, k, n, batch } = PROBE;
    const Xq = new Int8Array(batch * m * k), Wq = new Int8Array(batch * k * n);
    for (let i = 0; i < Xq.length; i++) Xq[i] = Math.round(rnd() * 254 - 127);
    for (let i = 0; i < Wq.length; i++) Wq[i] = Math.round(rnd() * 254 - 127);
    const rs = Float32Array.from({ length: batch * m }, () => 1);
    const cs = Float32Array.from({ length: batch * n }, () => 1);
    return { Xq, Wq, rs, cs, d: { ...PROBE, acc: true } };
  }
  // ---- transcendental probe ---------------------------------------------------
  // The kernel probe covers int8 GEMM arithmetic. It does NOT cover the JS
  // transcendentals, and ECMA-262 does not require Math.exp/log/cos/sin to be
  // correctly rounded — it calls them "implementation-approximated", so V8,
  // SpiderMonkey and JSC are all permitted to return different bits.
  //
  // One use of them can fork a fleet: weight INIT. Peers build their own
  // starting weights from a shared seed rather than broadcasting them
  // (Box-Muller, so Math.log/cos/sin), and one ulp of engine disagreement
  // means peers begin from different models. The per-step uses (softmax, loss)
  // cannot fork anything — every peer averages the same received gradient
  // BYTES, so a differently-rounded local gradient is still a gradient the
  // whole group then agrees on.
  //
  // This hashes the exact f64 bits of those four functions over a fixed grid,
  // so a browser whose math library differs is IDENTIFIED instead of showing
  // up later as an unexplained weight-hash mismatch. Cost: microseconds, once.
  // Detection, not correction — pinning the algorithms would be a fleet-wide
  // numerics change, and is only worth doing if this ever reports a mismatch.
  function mathProbe() {
    const buf = new ArrayBuffer(8), dv = new DataView(buf);
    let h = 0x811c9dc5;
    const eat = (x) => {
      dv.setFloat64(0, x);
      for (let i = 0; i < 8; i++) { h ^= dv.getUint8(i); h = Math.imul(h, 0x01000193); }
    };
    for (let i = 1; i <= 400; i++) {
      const u = i / 401;                     // (0,1): the Box-Muller domain
      eat(Math.log(u));
      eat(Math.cos(2 * Math.PI * u));
      eat(Math.sin(2 * Math.PI * u));
      eat(Math.sqrt(-2 * Math.log(u)));      // the exact composite randn uses
      eat(Math.exp(-u * 12));                // the softmax domain
    }
    return h >>> 0;
  }

  async function kernelProbe(compute, L) {
    const { Xq, Wq, rs, cs, d } = probeInputs();
    const out = compute && compute.bgemm
      ? await compute.bgemm(Xq, Wq, rs, cs, d)
      : root.Verified.bgemmJS(Xq, Wq, rs, cs, d, L);
    let h = 0x811c9dc5;                      // FNV-1a over the exact int32 results
    const b = new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
    for (let i = 0; i < b.length; i++) { h ^= b[i]; h = Math.imul(h, 0x01000193); }
    // fold in the transcendental hash: same question ("is your arithmetic the
    // same as mine?"), same broadcast slot, no wire-format change
    const mh = mathProbe();
    for (let i = 0; i < 4; i++) { h ^= (mh >>> (i * 8)) & 0xFF; h = Math.imul(h, 0x01000193); }
    return h >>> 0;
  }

  root.Compute = { initCompute, loadLUTs, kernelProbe, mathProbe };
})(self);
