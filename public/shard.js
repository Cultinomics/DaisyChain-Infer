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

  // Peers whose kernel probe disagrees with the head are not in the ring.
  // The head (index 0) is always kept. A missing probe is not a disagreement:
  // the hello has not arrived yet, and buildPlan only lists peers that sent one.
  function ringMembers(caps, probeHash) {
    if (!caps.length) return [];
    const head = caps[0];
    const rest = caps.slice(1).filter(d => !probeHash || !d.probe || d.probe === probeHash);
    return [head, ...rest];
  }

  // Budget check, not a latency model. `gemmPerSec` is the capacity probe:
  // how many 32x64x64 verified GEMMs this device finishes per second. The
  // count below is the MAC volume of one window through every layer, divided
  // by that unit. It is there to refuse a split that cannot produce a token
  // in `budgetMs`, before any weight download.
  function estimateLapMs(spec, T, gemmPerSec) {
    const unit = 32 * 64 * 64;
    const C = spec.hidden;
    const inter = spec.inter || (C * 4);
    const hd = spec.headDim || Math.floor(C / spec.heads);
    const q = spec.heads * hd;
    const kv = (spec.kvHeads || spec.heads) * hd;
    const macPerLayer = T * C * q + 2 * T * C * kv + T * q * C + 3 * T * C * inter
      + spec.heads * (T * hd * T + T * T * hd);
    const gemms = (macPerLayer * spec.layers) / unit;
    return gemms / Math.max(gemmPerSec, 1e-6) * 1000;
  }

  // Whole-model f32 bytes versus a fraction of reported device RAM.
  // `deviceGB` comes from navigator.deviceMemory when the browser provides it.
  // Unknown memory returns false: the ring is used, rather than guessing.
  function headCanHold(spec, tensors, deviceGB) {
    if (!deviceGB || deviceGB <= 0) return false;
    let bytes = 0;
    for (const t of tensors.values()) bytes += t.elems * 4;
    return bytes < deviceGB * 0.25 * 1073741824;
  }

  // A short prompt stays on the head when the head can hold every layer.
  // The ring still exists for prompts that do not fit that rule.
  function preferLocal(promptTokens, peerCount, fits) {
    return !!(fits && peerCount > 0 && promptTokens > 0 && promptTokens <= 64);
  }

  // Rolling FNV blocks over token ids. This is the approximate prefix index:
  // it records which prefix this chat last ran. The forward pass still
  // recomputes the window. Per-column value scales depend on every token in
  // the window, so skipping a matched block would not be bit-identical to the
  // full pass, and this project does not ship a second answer.
  function makePrefixIndex(blockSize) {
    const B = blockSize || 64;
    const sessions = new Map();
    function blocks(ids) {
      const out = [];
      for (let i = 0; i < ids.length; i += B) {
        const slice = ids.slice(i, Math.min(ids.length, i + B));
        out.push(fnv1a(new TextEncoder().encode(slice.join(","))));
      }
      return out;
    }
    return {
      blockSize: B,
      blocks,
      note(session, ids) { sessions.set(session, blocks(ids)); },
      matchedBlocks(session, ids) {
        const prev = sessions.get(session);
        if (!prev) return 0;
        const now = blocks(ids);
        let n = 0;
        while (n < now.length && n < prev.length && now[n] === prev[n]) n++;
        return n;
      },
    };
  }

  // In-memory KV spill. When the byte cap is hit, the oldest block is dropped
  // and the caller recomputes that span. `held` is what remained.
  function makeKvSpill(maxBytes) {
    let used = 0;
    const order = [];
    const map = new Map();
    function drop() {
      const k = order.shift();
      const old = map.get(k);
      if (!old) return;
      map.delete(k);
      used -= old.byteLength;
    }
    return {
      get used() { return used; },
      get size() { return map.size; },
      put(key, bytes) {
        const n = bytes.byteLength;
        if (map.has(key)) {
          used -= map.get(key).byteLength;
          const i = order.indexOf(key);
          if (i >= 0) order.splice(i, 1);
        }
        while (order.length && used + n > maxBytes) drop();
        if (n > maxBytes) return false;
        map.set(key, bytes);
        order.push(key);
        used += n;
        return true;
      },
      get(key) { return map.get(key) || null; },
    };
  }

  // Prompt chunks for a pipelined prefill. `window` is how many chunks may be
  // in flight across the ring at once. Decode stays one token, one lap:
  // the next token is not known until the lap returns.
  function prefillChunks(nTokens, chunkSize, inFlight) {
    const size = Math.max(1, chunkSize | 0);
    const window = Math.max(1, inFlight | 0);
    const chunks = [];
    for (let lo = 0; lo < nTokens; lo += size)
      chunks.push({ index: chunks.length, lo, hi: Math.min(nTokens, lo + size) });
    return { chunks, window, chunkSize: size };
  }

  // Prompt-lookup draft: the longest suffix of the token ids that already
  // occurred earlier, and the tokens that followed it. No second model.
  // Empty when nothing repeats. The ring then scores that span in one lap.
  function promptLookupDraft(ids, maxDraft) {
    const cap = Math.max(0, maxDraft | 0);
    const n = ids.length;
    if (cap === 0 || n < 3) return [];
    let best = [], bestLen = 0;
    const maxLen = Math.min(n - 1, 8);
    for (let len = 2; len <= maxLen; len++) {
      for (let i = 0; i + len < n; i++) {
        let match = true;
        for (let j = 0; j < len; j++) if (ids[i + j] !== ids[n - len + j]) { match = false; break; }
        if (!match) continue;
        const draft = [];
        // Tokens after the earlier copy, and before the suffix we just matched.
        for (let k = 0; k < cap && i + len + k < n - len; k++) draft.push(ids[i + len + k]);
        if (!draft.length) continue;
        if (len > bestLen) { best = draft; bestLen = len; }
      }
    }
    return best;
  }

  // picked[i] proposes draft[i]. picked has one extra entry: the token at the
  // first position that rejected, which is always kept.
  function acceptDraft(picked, draft) {
    const out = [];
    let i = 0;
    while (i < draft.length && i < picked.length && picked[i] === draft[i]) out.push(draft[i]), i++;
    if (i < picked.length) out.push(picked[i]);
    return out;
  }

  const api = { planStages, stageBytes, stageWeights, toKN, transposeEmb,
                fnv1a, hashF32, modelFingerprint, ringMembers, estimateLapMs,
                headCanHold, preferLocal, makePrefixIndex, makeKvSpill,
                prefillChunks, promptLookupDraft, acceptDraft };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Shard = api;
})(typeof self !== "undefined" ? self : this);
