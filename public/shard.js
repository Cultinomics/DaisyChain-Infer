// Model sharding — the piece DaisyChain-Train does not have.
//
// DaisyChain-Train pools COMPUTE: every node holds a full replica, so a model
// bigger than one machine cannot be trained. Inference is where that limit can
// be lifted, because a forward pass is a chain: layer l needs layer l-1's
// OUTPUT, never its WEIGHTS. So the layers live on different machines and the
// activation travels instead.
//
// And because safetensors gives every tensor an exact byte range, each device
// fetches only its own layers straight from the Hub. No device — not even the
// head — ever holds the whole model. That is what makes the pooling real
// rather than a redistribution of something one machine already had to load.
//
// The ring shape is forced by weight tying. When lm_head is tied to the
// embedding table, the largest tensor in the model is needed at BOTH ends: to
// embed the prompt and to produce the logits. Copying it to the last stage
// would hand back most of the memory just pooled, so the last stage returns
// its hidden state to the head, which owns the embedding once and does both
// ends of the pass.
(function (root) {
  "use strict";

  // ---- the plan ---------------------------------------------------------------
  // Stage 0 is the HEAD: embeddings, final norm, lm_head. Every stage may also
  // own a contiguous run of layers, apportioned by MEASURED capacity — the
  // same self-calibrating idea as cluster.py's capacity_score, except what is
  // balanced is layers per device rather than batch per device.
  //
  // Largest-remainder apportionment, so the plan is a pure function of the
  // capacity report: every device derives the same plan from the same inputs
  // and the plan never has to be trusted, only compared.
  function planStages(spec, caps) {
    const n = caps.length;
    if (n < 1) throw new Error("no devices to plan across");
    const total = caps.reduce((a, d) => a + Math.max(1e-6, d.capacity), 0);
    const exact = caps.map(d => spec.layers * Math.max(1e-6, d.capacity) / total);
    const floor = exact.map(Math.floor);
    let left = spec.layers - floor.reduce((a, b) => a + b, 0);
    const order = exact.map((e, i) => [e - floor[i], i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    for (let i = 0; i < order.length && left > 0; i++, left--) floor[order[i][1]]++;
    const stages = [];
    let lo = 0;
    for (let i = 0; i < n; i++) {
      const hi = lo + floor[i];
      stages.push({ id: caps[i].id, index: i, lo, hi, head: i === 0,
                    backend: caps[i].backend, capacity: caps[i].capacity });
      lo = hi;
    }
    // A device with no layers is only worth a hop if it is the head, which has
    // real work (embed + unembed). Anyone else empty is dropped.
    return stages.filter(s => s.head || s.hi > s.lo).map((s, i) => ({ ...s, index: i }));
  }

  // What a stage will cost to hold, computed from the file's own header BEFORE
  // any weight bytes move — so a device can see whether its slice fits before
  // spending the bandwidth finding out.
  function stageBytes(spec, available, st, ArchMod) {
    const names = ArchMod.tensorsFor(spec, available, st);
    let n = 0;
    for (const name of names) n += available.get(name).elems * 4;   // f32 in memory
    return n;
  }

  // ---- arranging fetched tensors into what the forward pass wants -------------
  // Two shape conventions have to be reconciled here, and getting it wrong is
  // silent: torch.nn.Linear stores weights (out, in) while the GEMM wants
  // (in, out), whereas GPT-2's Conv1D already stores (in, out). A wrong
  // transpose does not throw — it produces a model that generates confident
  // nonsense — so the layout comes from the spec and is applied once, at load.
  function toKN(w, outDim, inDim, layout) {
    if (layout === "in_out") return w;                    // already k x n
    const out = new Float32Array(w.length);
    for (let o = 0; o < outDim; o++)
      for (let i = 0; i < inDim; i++) out[i * outDim + o] = w[o * inDim + i];
    return out;
  }

  function stageWeights(spec, st, got, ArchMod) {
    const resolve = ArchMod.resolver(new Map([...got.keys()].map(k => [k, true])));
    const pick = (name, opt) => { const r = resolve(name, opt); return r ? got.get(r) : null; };
    const C = spec.hidden, L = spec.weightLayout;
    const w = { layers: [] };

    if (st.head) {
      const h = ArchMod.headTensors(spec);
      w.emb = pick(h.emb);                                // (vocab, C) — used as a lookup, not a GEMM
      w.pos = pick(h.pos, true);
      w.nrmF = pick(h.nrmF, true);
      w.nrmFb = pick(h.nrmFb, true);
      // lm_head as k x n = (C, vocab). When tied, that is embᵀ.
      const lm = pick(h.lmHead, true);
      w.lmHead = lm ? toKN(lm, spec.vocab, C, L) : transposeEmb(w.emb, spec.vocab, C);
      if (!w.nrmF) w.nrmF = new Float32Array(C).fill(1);  // a model without a final norm weight
    }

    const qDim = spec.heads * spec.headDim, kvDim = spec.kvHeads * spec.headDim;
    for (let l = st.lo; l < st.hi; l++) {
      const t = ArchMod.layerTensors(spec, l);
      const ly = { nrm1: pick(t.nrm1), nrm1b: pick(t.nrm1b, true),
                   nrm2: pick(t.nrm2), nrm2b: pick(t.nrm2b, true) };
      if (spec.qkvFused) {
        ly.Wqkv = toKN(pick(t.Wqkv), 3 * C, C, L);
        ly.bqkv = pick(t.bqkv, true);
      } else {
        ly.Wq = toKN(pick(t.Wq), qDim, C, L); ly.bq = pick(t.bq, true);
        ly.Wk = toKN(pick(t.Wk), kvDim, C, L); ly.bk = pick(t.bk, true);
        ly.Wv = toKN(pick(t.Wv), kvDim, C, L); ly.bv = pick(t.bv, true);
      }
      ly.Wo = toKN(pick(t.Wo), C, qDim, L); ly.bo = pick(t.bo, true);
      if (spec.gated) {
        ly.Wgate = toKN(pick(t.Wgate), spec.inter, C, L);
        ly.Wup = toKN(pick(t.Wup), spec.inter, C, L);
      } else {
        ly.Wfc = toKN(pick(t.Wfc), spec.inter, C, L); ly.bfc = pick(t.bfc, true);
      }
      ly.Wdown = toKN(pick(t.Wdown), C, spec.inter, L); ly.bdown = pick(t.bdown, true);
      w.layers.push(ly);
    }
    return w;
  }
  function transposeEmb(emb, vocab, C) {
    const out = new Float32Array(emb.length);
    for (let v = 0; v < vocab; v++)
      for (let c = 0; c < C; c++) out[c * vocab + v] = emb[v * C + c];
    return out;
  }

  // ---- hashes -----------------------------------------------------------------
  // FNV-1a over raw bytes, identical to the function DaisyChain-Web hashes
  // replicas with, so hashes stay comparable across the projects.
  function fnv1a(bytes) {
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193); }
    return h >>> 0;
  }
  function hashF32(a) { return fnv1a(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)); }

  // The model fingerprint every stage repeats in its status. Derived from the
  // repo id, revision and the tensor index — NOT from the weights, because no
  // device reads all of them. It answers "are we all running the same model?",
  // which is the question that matters when stages hold disjoint pieces.
  function modelFingerprint(repo, revision, tensors) {
    const parts = [repo, revision || "main"];
    for (const name of [...tensors.keys()].sort()) {
      const t = tensors.get(name);
      parts.push(`${name}:${t.dtype}:${t.shape.join("x")}:${t.start}:${t.end}`);
    }
    return fnv1a(new TextEncoder().encode(parts.join("|")));
  }

  const api = { planStages, stageBytes, stageWeights, toKN, transposeEmb,
                fnv1a, hashF32, modelFingerprint };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Shard = api;
})(typeof self !== "undefined" ? self : this);
