// The central claim of this project, as a test that runs.
//
// Splitting a model across machines is only safe if the split changes nothing.
// That is checkable here in a way it usually is not, because the pieces are
// deterministic: run a prompt with every layer on one "device", then again
// with the layers handed out across N simulated stages passing hidden states
// between them, and compare.
//
// The comparison is EXACT — bit patterns, not a tolerance. A correct split is
// not merely close to the unsplit result, it is identical, because the ring
// never re-derives anything; it only moves f32 arrays. A tolerance here would
// hide exactly the bugs worth finding: dropping one layer out of twelve moves
// the logits far less than you would guess, and still reads as fluent text.
//
//   node test_pipeline.js
const assert = require("assert");
const Shard = require("./public/shard.js");
const Infer = require("./public/infer.js");
const Arch = require("./public/arch.js");

// The LUT is exactly a*b for int8 — what the verified mul8 unit is proven to
// reproduce. Rebuilding it here keeps the test fixture-free without weakening
// it, since the unit's whole claim is that it equals this.
const L = (() => {
  const mul = new Int16Array(65536);
  for (let a = 0; a < 256; a++)
    for (let b = 0; b < 256; b++) mul[a * 256 + b] = (a > 127 ? a - 256 : a) * (b > 127 ? b - 256 : b);
  return { mul };
})();
const ctx = () => ({ L, bgemm: null, att: null, mlp: null, audit: null });   // CPU mirror

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function rnd(n, seed, scale) {
  const r = mulberry32(seed), o = new Float32Array(n);
  for (let i = 0; i < n; i++) o[i] = (r() * 2 - 1) * (scale || 1);
  return o;
}

// Build a weights map keyed exactly as the real repos key them, so the test
// exercises the same name resolution and layout handling the loader uses.
function makeWeights(spec) {
  const got = new Map();
  const C = spec.hidden, qDim = spec.heads * spec.headDim, kvDim = spec.kvHeads * spec.headDim;
  let seed = 7;
  const put = (n, len, scale) => got.set(n, rnd(len, seed++, scale ?? 0.08));
  const h = Arch.headTensors(spec);
  put(h.emb, spec.vocab * C, 0.05);
  if (h.pos) put(h.pos, spec.maxPos * C, 0.02);
  put(h.nrmF, C, 1);
  got.set(h.nrmF, new Float32Array(C).fill(1));
  if (spec.norm === "ln") got.set(h.nrmFb, new Float32Array(C));
  if (!spec.tie) put(h.lmHead, spec.vocab * C, 0.05);
  for (let l = 0; l < spec.layers; l++) {
    const t = Arch.layerTensors(spec, l);
    got.set(t.nrm1, new Float32Array(C).fill(1));
    got.set(t.nrm2, new Float32Array(C).fill(1));
    if (spec.norm === "ln") { got.set(t.nrm1b, new Float32Array(C)); got.set(t.nrm2b, new Float32Array(C)); }
    if (spec.qkvFused) { put(t.Wqkv, C * 3 * C); got.set(t.bqkv, new Float32Array(3 * C)); }
    else { put(t.Wq, qDim * C); put(t.Wk, kvDim * C); put(t.Wv, kvDim * C); }
    put(t.Wo, C * qDim);
    if (spec.qkvFused) got.set(t.bo, new Float32Array(C));
    if (spec.gated) { put(t.Wgate, spec.inter * C); put(t.Wup, spec.inter * C); }
    else { put(t.Wfc, spec.inter * C); got.set(t.bfc, new Float32Array(spec.inter)); }
    put(t.Wdown, C * spec.inter);
    if (!spec.gated) got.set(t.bdown, new Float32Array(C));
  }
  return got;
}

function stagesFor(spec, got, splits, T) {
  const plan = [];
  let lo = 0;
  splits.forEach((n, i) => { plan.push({ id: "p" + i, index: i, lo, hi: lo + n, head: i === 0 }); lo += n; });
  assert.strictEqual(lo, spec.layers, "splits must cover every layer exactly once");
  return plan.map(st => Infer.makeStage(spec, st, Shard.stageWeights(spec, st, got, Arch), ctx(), T));
}

async function ringGenerate(spec, stages, ids, nTok, T) {
  const head = stages[0];
  const out = [...ids];
  for (let n = 0; n < nTok; n++) {
    const win = new Int32Array(T);
    const tail = out.slice(-T);
    for (let i = 0; i < tail.length; i++) win[T - tail.length + i] = tail[i];
    let x = Infer.embed(head, win);
    for (const S of stages) x = await Infer.runLayers(S, x);       // the "hop"
    out.push(Infer.pickToken(await Infer.readout(head, x), { temperature: 0 }));
  }
  return out;
}

function bitsEqual(a, b) {
  if (a.length !== b.length) return false;
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
}

// Two architectures, because the whole point of arch.js is that the ring does
// not care which one it is running. Llama-style exercises RMSNorm/RoPE/GQA/
// SwiGLU; GPT-2-style exercises LayerNorm+bias, fused QKV and GELU.
const LLAMA = Arch.fromConfig({
  model_type: "llama", hidden_size: 64, num_hidden_layers: 6, num_attention_heads: 4,
  num_key_value_heads: 2, intermediate_size: 128, vocab_size: 128,
  max_position_embeddings: 64, rms_norm_eps: 1e-5, tie_word_embeddings: true,
});
const GPT2 = Arch.fromConfig({
  model_type: "gpt2", n_embd: 64, n_layer: 6, n_head: 4, vocab_size: 128,
  n_positions: 64, n_inner: 128, layer_norm_epsilon: 1e-5,
});

(async function () {
  let failed = 0;
  const t = (name, fn) => fn().then(
    () => console.log(`  ok    ${name}`),
    (e) => { failed++; console.log(`  FAIL  ${name}\n        ${e.message}`); });

  console.log("\nDaisyChain-Infer — pipeline equivalence\n");

  const T = 16, PROMPT = [5, 11, 42, 7];

  for (const [label, spec] of [["llama-style", LLAMA], ["gpt2-style", GPT2]]) {
    console.log(`  ${label}: ${spec.layers} layers, hidden ${spec.hidden}, ` +
                `heads ${spec.heads}/${spec.kvHeads}, ${spec.gated ? "SwiGLU" : "GELU"}, ` +
                `${spec.norm.toUpperCase()}, rope=${spec.rope}`);
    const got = makeWeights(spec);
    const ref = await ringGenerate(spec, stagesFor(spec, got, [spec.layers], T), PROMPT, 8, T);

    await t(`${label}: a stage boundary changes no bit of the hidden state`, async () => {
      const whole = stagesFor(spec, got, [spec.layers], T);
      const split = stagesFor(spec, got, [2, 2, 2], T);
      const win = new Int32Array(T);
      for (let i = 0; i < PROMPT.length; i++) win[T - PROMPT.length + i] = PROMPT[i];
      const a = await Infer.runLayers(whole[0], Infer.embed(whole[0], win));
      let b = Infer.embed(split[0], win);
      for (const S of split) b = await Infer.runLayers(S, b);
      assert.ok(bitsEqual(a, b), "hidden states diverged across the split");
    });

    for (const splits of [[3, 3], [1, 2, 3], [0, 3, 3], [2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]) {
      await t(`${label}: generation is identical when split ${JSON.stringify(splits)}`, async () => {
        const out = await ringGenerate(spec, stagesFor(spec, got, splits, T), PROMPT, 8, T);
        assert.deepStrictEqual(out, ref, `split ${splits} produced different tokens`);
      });
    }

    // Mutation checks. Without these, a test where both sides call the same
    // code proves nothing at all.
    await t(`${label}: a stage that silently skips a layer IS caught`, async () => {
      const stages = stagesFor(spec, got, [2, 2, 2], T);
      stages[1].w.layers.pop();
      const out = await ringGenerate(spec, stages, PROMPT, 8, T);
      assert.notDeepStrictEqual(out, ref, "dropping a whole layer changed nothing — the check is blind");
    });
    await t(`${label}: stages applied out of order ARE caught`, async () => {
      const stages = stagesFor(spec, got, [2, 2, 2], T);
      const out = await ringGenerate(spec, [stages[0], stages[2], stages[1]], PROMPT, 8, T);
      assert.notDeepStrictEqual(out, ref, "layer order did not matter — the check is blind");
    });
  }

  // A wrong transpose does not throw; it yields a model that generates
  // confident nonsense. So the layout conversion gets its own check.
  await t("weight layout conversion is a real transpose, and in_out is a no-op", async () => {
    const w = Float32Array.from([1, 2, 3, 4, 5, 6]);           // (out=2, in=3)
    const kn = Shard.toKN(w, 2, 3, "out_in");
    assert.deepStrictEqual([...kn], [1, 4, 2, 5, 3, 6], "out_in did not transpose to k x n");
    assert.strictEqual(Shard.toKN(w, 2, 3, "in_out"), w, "in_out must pass through untouched");
  });

  await t("architecture detection maps real configs, and refuses unknown ones", async () => {
    assert.strictEqual(Arch.fromConfig({ model_type: "qwen2", hidden_size: 896, num_hidden_layers: 24,
      num_attention_heads: 14, num_key_value_heads: 2, intermediate_size: 4864, vocab_size: 151936 }).family, "llama");
    assert.strictEqual(Arch.fromConfig({ model_type: "gpt2", n_embd: 768, n_layer: 12,
      n_head: 12, vocab_size: 50257 }).family, "gpt2");
    assert.throws(() => Arch.fromConfig({ model_type: "mamba", hidden_size: 8, num_hidden_layers: 1,
      num_attention_heads: 1, vocab_size: 8 }), /unsupported architecture/);
    assert.throws(() => Arch.fromConfig({ model_type: "llama" }), /missing/);
  });

  await t("planning is deterministic and capacity-weighted", async () => {
    const caps = [{ id: "p1", capacity: 100 }, { id: "p2", capacity: 300 }, { id: "p3", capacity: 200 }];
    const a = Shard.planStages(LLAMA, caps), b = Shard.planStages(LLAMA, caps);
    assert.deepStrictEqual(a, b, "planning is not deterministic");
    assert.strictEqual(a.reduce((s, x) => s + (x.hi - x.lo), 0), LLAMA.layers, "plan does not cover every layer");
    assert.ok(a[1].hi - a[1].lo > a[0].hi - a[0].lo, "the faster device did not get more layers");
    assert.ok(a[0].head, "stage 0 must be the head");
  });

  await t("a disagreeing kernel probe is left out of the ring", async () => {
    const caps = [
      { id: "head", capacity: 100, probe: 1 },
      { id: "ok", capacity: 100, probe: 1 },
      { id: "bad", capacity: 900, probe: 2 },
    ];
    const members = Shard.ringMembers(caps, 1);
    assert.deepStrictEqual(members.map(d => d.id), ["head", "ok"]);
    const plan = Shard.planStages(LLAMA, members);
    assert.ok(plan.every(s => s.id !== "bad"));
    assert.strictEqual(plan.reduce((s, x) => s + (x.hi - x.lo), 0), LLAMA.layers);
  });

  await t("lap estimate grows with context and refuses an idle device", async () => {
    const spec = Object.assign({}, LLAMA, { inter: 64, headDim: 8 });
    const slow = Shard.estimateLapMs(spec, 128, 0.01);
    const fast = Shard.estimateLapMs(spec, 8, 1000);
    assert.ok(slow > fast, "a longer window on a slower device must cost more");
    assert.ok(fast > 0);
    const solo = Shard.estimatePlanMs(spec, 32, [{ lo: 0, hi: spec.layers, head: true, capacity: 1000 }]);
    const ring = Shard.estimatePlanMs(spec, 32, [
      { lo: 0, hi: 3, head: true, capacity: 1000 },
      { lo: 3, hi: spec.layers, head: false, capacity: 10 },
    ]);
    assert.ok(ring > solo, "a slow stage must raise the lap estimate");
  });

  await t("a short prompt stays local only when the head can hold the model", async () => {
    const tensors = new Map([["w", { elems: 1000 }]]);
    const spec = { hidden: 8 };
    assert.strictEqual(Shard.headCanHold(spec, tensors, 0), false);
    assert.strictEqual(Shard.headCanHold(spec, tensors, 8), true);
    assert.strictEqual(Shard.preferLocal(16, 2, true), true);
    assert.strictEqual(Shard.preferLocal(128, 2, true), true);
    assert.strictEqual(Shard.preferLocal(16, 2, false), false);
    assert.strictEqual(Shard.preferLocal(16, 0, true), false);
  });

  await t("prefix blocks match a continuation and the spill evicts the oldest block", async () => {
    const idx = Shard.makePrefixIndex(4);
    const a = [1, 2, 3, 4, 5, 6, 7, 8];
    const kept = [1, 2, 3, 4, 5];
    idx.restore("kept", idx.blocks(kept).slice(0, 1));
    assert.strictEqual(idx.matchedBlocks("kept", kept.concat([6])), 1);
    idx.note("chat", a);
    idx.note("other", [9, 9, 9, 9, 8]);
    assert.strictEqual(idx.matchedBlocks("chat", a.concat([9])), 2);
    assert.strictEqual(idx.matchedBlocks("other", [9, 9, 9, 9, 1]), 1);
    assert.strictEqual(idx.matchedBlocks("chat", [9, 9, 9, 9]), 0);
    const spill = Shard.makeKvSpill(12);
    assert.strictEqual(spill.put("a", new Uint8Array(8)), true);
    assert.strictEqual(spill.put("b", new Uint8Array(8)), true);
    assert.strictEqual(spill.get("a"), null, "the oldest block must be spilled");
    assert.ok(spill.get("b"));
    assert.strictEqual(spill.put("huge", new Uint8Array(13)), false);
  });

  await t("prefill chunks, prompt-lookup drafts, and accept keep a bonus token", async () => {
    const c = Shard.prefillChunks(40, 16, 2);
    assert.strictEqual(c.chunks.length, 3);
    assert.deepStrictEqual(c.chunks.map(x => x.hi - x.lo), [14, 13, 13]);
    assert.strictEqual(c.chunks[2].hi, 40);
    assert.strictEqual(c.chunks[0].lo, 0);
    const frame = Shard.lookaheadFrame([1, 2, 3, 9, 1, 2, 3], 16, 4);
    assert.deepStrictEqual(frame.draft, [9]);
    assert.strictEqual(frame.win.length, 8);
    assert.strictEqual(c.window, 2);
    const ids = [1, 2, 3, 9, 1, 2, 3];
    assert.deepStrictEqual(Shard.promptLookupDraft(ids, 4), [9]);
    assert.deepStrictEqual(Shard.acceptDraft([9, 4], [9]), [9, 4]);
    assert.deepStrictEqual(Shard.acceptDraft([8, 4], [9]), [8]);
  });

  await t("chunked prefill matches across a stage split", async () => {
    const got = makeWeights(LLAMA);
    const prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
    const whole = await Infer.generateChunked(stagesFor(LLAMA, got, [LLAMA.layers], 16), prompt, 3, 4, { temperature: 0 });
    const split = await Infer.generateChunked(stagesFor(LLAMA, got, [2, 2, 2], 16), prompt, 3, 4, { temperature: 0 });
    assert.deepStrictEqual(split, whole);
  });

  await t("a lookahead draft scores in one window and survives a split", async () => {
    const got = makeWeights(LLAMA);
    const prompt = [4, 4, 4, 7, 4, 4, 4];
    const opts = { temperature: 0, maxDraft: 4 };
    const whole = await Infer.generateLookahead(stagesFor(LLAMA, got, [LLAMA.layers], 16), prompt, 4, 16, opts);
    const split = await Infer.generateLookahead(stagesFor(LLAMA, got, [1, 2, 3], 16), prompt, 4, 16, opts);
    assert.deepStrictEqual(split, whole);
    assert.strictEqual(whole.length, prompt.length + 4);
  });

  await t("a parameter shard matches a resident forward and keeps at most 2 layers live", async () => {
    for (const spec of [LLAMA, GPT2]) {
      const got = makeWeights(spec);
      const st = { id: "p0", index: 0, lo: 0, hi: spec.layers, head: true };
      const wRes = Shard.stageWeights(spec, st, got, Arch);
      const wSh = Shard.stageWeights(spec, st, got, Arch);
      const resident = Infer.makeStage(spec, st, wRes, ctx(), T, { resident: true });
      const sharded = Infer.makeStage(spec, st, wSh, ctx(), T);
      assert.ok(sharded.paramShard, "a multi-layer stage should flat-shard");
      assert.strictEqual(sharded.paramShard.limit, 2);
      assert.strictEqual(resident.paramShard, null);
      assert.ok(wSh.emb && wSh.emb.length, "embeddings stay outside the shard");
      assert.ok(wSh.lmHead && wSh.lmHead.length, "lm_head stays resident");
      const firstKey = sharded.paramShard.slices[0][0].k;
      assert.strictEqual(wSh.layers[0][firstKey], null, "weights are packed, not left on the layer");
      const win = new Int32Array(T);
      for (let i = 0; i < PROMPT.length; i++) win[T - PROMPT.length + i] = PROMPT[i];
      const a = await Infer.runLayers(resident, Infer.embed(resident, win));
      const flatBefore = sharded.paramShard.flat.slice();
      const b = await Infer.runLayers(sharded, Infer.embed(sharded, win));
      assert.ok(bitsEqual(a, b), "sharded runLayers diverged from resident");
      assert.strictEqual(sharded.paramShard.live, 0);
      assert.strictEqual(sharded.paramShard.maxLive, 2);
      assert.strictEqual(wSh.layers[0][firstKey], null);
      assert.strictEqual(wSh.layers[spec.layers - 1][firstKey], null);
      assert.ok(bitsEqual(flatBefore, sharded.paramShard.flat), "the forward wrote the flat shard");
      for (let i = 0; i < wRes.layers.length; i++) {
        for (const p of sharded.paramShard.slices[i]) {
          const src = wRes.layers[i][p.k];
          assert.strictEqual(src.length, p.len);
          for (let j = 0; j < p.len; j++)
            if (sharded.paramShard.flat[p.off + j] !== src[j])
              throw new Error("flat shard byte mismatch at layer " + i);
        }
      }
      const ids = [1, 2, 3, 4];
      const spanR = { pos0: 0, cache: [] };
      const spanS = { pos0: 0, cache: [] };
      const cr = await Infer.runLayersChunk(resident, Infer.embedRows(resident, ids, 0), spanR);
      const cs = await Infer.runLayersChunk(sharded, Infer.embedRows(sharded, ids, 0), spanS);
      assert.ok(bitsEqual(cr, cs), "sharded chunk diverged from resident");
      assert.ok(bitsEqual(spanR.cache[0].k, spanS.cache[0].k));
      assert.ok(bitsEqual(spanR.cache[0].v, spanS.cache[0].v));
      assert.strictEqual(sharded.paramShard.live, 0);
      assert.ok(bitsEqual(flatBefore, sharded.paramShard.flat));
      Shard.gatherLayer(sharded.paramShard, wSh.layers, 0);
      Shard.gatherLayer(sharded.paramShard, wSh.layers, 1);
      assert.throws(() => Shard.gatherLayer(sharded.paramShard, wSh.layers, 2), /2 live/);
      Shard.releaseLayer(sharded.paramShard, wSh.layers, 0);
      Shard.releaseLayer(sharded.paramShard, wSh.layers, 1);
      assert.strictEqual(sharded.paramShard.live, 0);
    }
    const st1 = { id: "p0", index: 0, lo: 0, hi: 1, head: true };
    const one = Infer.makeStage(LLAMA, st1, Shard.stageWeights(LLAMA, st1, makeWeights(LLAMA), Arch), ctx(), T);
    assert.strictEqual(one.paramShard, null, "a single layer stays resident");
    assert.ok(one.w.layers[0].Wq && one.w.layers[0].Wq.length);
    assert.strictEqual(Shard.makeParamShard([one.w.layers[0]]), null);
  });

  await t("sliding-window mask matches across a split and stays full-width", async () => {
    const got = makeWeights(LLAMA);
    const win = new Int32Array(T);
    for (let i = 0; i < PROMPT.length; i++) win[T - PROMPT.length + i] = PROMPT[i];
    const run = async (spec, splits) => {
      const stages = stagesFor(spec, got, splits, T);
      let x = Infer.embed(stages[0], win);
      for (const S of stages) x = await Infer.runLayers(S, x);
      return x;
    };
    const plain = await run(LLAMA, [LLAMA.layers]);
    const wide = Object.assign({}, LLAMA, { slidingWindow: 1000, swaFrom: 0 });
    assert.ok(bitsEqual(plain, await run(wide, [LLAMA.layers])), "a window wider than the cache changed the GEMM");
    const late = Object.assign({}, LLAMA, { slidingWindow: 2, swaFrom: LLAMA.layers });
    assert.ok(bitsEqual(plain, await run(late, [LLAMA.layers])), "layers before swaFrom must stay full causal");
    const sw = Object.assign({}, LLAMA, { slidingWindow: 2, swaFrom: 0 });
    const masked = await run(sw, [sw.layers]);
    assert.ok(!bitsEqual(plain, masked), "a window of 2 did not change attention");
    const split = await run(sw, [2, 2, 2]);
    assert.ok(bitsEqual(masked, split), "sliding-window split diverged");
    const cfg = Arch.fromConfig({
      model_type: "qwen2", hidden_size: 64, num_hidden_layers: 4, num_attention_heads: 4,
      num_key_value_heads: 2, intermediate_size: 128, vocab_size: 32,
      use_sliding_window: true, sliding_window: 8, max_window_layers: 2,
    });
    assert.strictEqual(cfg.slidingWindow, 8);
    assert.strictEqual(cfg.swaFrom, 2);
    const off = Arch.fromConfig({
      model_type: "qwen2", hidden_size: 64, num_hidden_layers: 4, num_attention_heads: 4,
      num_key_value_heads: 2, intermediate_size: 128, vocab_size: 32,
      use_sliding_window: false, sliding_window: 8,
    });
    assert.strictEqual(off.slidingWindow, 0);
  });

  await t("JSON mask, position seed, and a prepared prompt", async () => {
    assert.strictEqual(Shard.jsonStatus("{\"a\":1}"), "ok");
    assert.strictEqual(Shard.jsonAllows("", "{"), true);
    assert.strictEqual(Shard.jsonAllows("{", "a"), false);
    assert.strictEqual(Shard.jsonAllows("{", "\"a\":"), true);
    assert.strictEqual(Shard.jsonAllows("{\"a\":1}", "x"), false);
    assert.strictEqual(Shard.jsonAllows("tru", "e"), true);
    assert.strictEqual(Shard.jsonAllows("tru", "k"), false);
    assert.strictEqual(Shard.jsonAllows("[1,", "2]"), true);
    assert.strictEqual(Shard.jsonAllows("{\"a\":1,", "}"), false);
    assert.strictEqual(Shard.jsonAllows("", " "), true);
    const logits = Float32Array.from([5, 0.1, 1]);
    const decodePiece = (id) => ["x", "{", "}"][id];
    assert.strictEqual(Infer.pickStructured(logits, {
      schema: "json", temperature: 0, topK: 3, produced: "", decodePiece,
    }), 1);
    assert.strictEqual(Infer.pickStructured(logits, { temperature: 0 }), 0);
    const sample = Float32Array.from([0, 0.2, 0.5, 1.2, 0.1]);
    const a = Infer.pickToken(sample, { temperature: 0.8, topK: 5, seqSeed: 99, pos: 4, rng: () => 0.1 });
    const b = Infer.pickToken(sample, { temperature: 0.8, topK: 5, seqSeed: 99, pos: 4, rng: () => 0.9 });
    assert.strictEqual(a, b, "a position seed must ignore the mutable rng");
    assert.strictEqual(Infer.seqDraw(99, 4), Infer.seqDraw(99, 4));
    assert.deepStrictEqual(Shard.preparedIds("hi", { text: "hi", ids: [1, 2] }), [1, 2]);
    assert.strictEqual(Shard.preparedIds("no", { text: "hi", ids: [1] }), null);
    const gotIds = Shard.preparedIds("hi", { text: "hi", ids: [1, 2] });
    gotIds.push(9);
    assert.deepStrictEqual(Shard.preparedIds("hi", { text: "hi", ids: [1, 2] }), [1, 2]);
  });

  await t("a 128-channel weight scale is not the verified column scale", async () => {
    // DeepGEMM's FP8 GEMM scales every 128 channels of K (BLOCK_K == 128).
    // vgemmBlock uses one scale for the whole column. A spike in a later
    // block must not tighten the earlier block under our recipe, and it
    // must widen the single column scale. The forward path does not call
    // the block recipe.
    const V = require("./public/verified_core.js");
    const k = 256, n = 3, block = 128;
    const W = new Float32Array(k * n);
    for (let i = 0; i < W.length; i++) W[i] = ((i * 17) % 50) / 25 - 1;
    W[200 * n] = 40;
    const col = V.quantizeCols(W, k, n);
    assert.strictEqual(col.s.length, n);
    const blockS = new Float32Array((k / block) * n);
    for (let c = 0; c < n; c++) {
      for (let b = 0; b < k / block; b++) {
        let mx = 0;
        for (let r = b * block; r < (b + 1) * block; r++) mx = Math.max(mx, Math.abs(W[r * n + c]));
        blockS[b * n + c] = Math.max(mx / 127, 1e-8);
      }
    }
    assert.ok(blockS[0] < col.s[0], "the early block must not inherit a later spike");
    assert.strictEqual(col.s[0], Math.fround(40 / 127));
    assert.strictEqual(V.MAX_INT32_K, 131071);
  });

  await t("a non-finite activation is refused before any layer runs", async () => {
    const S = { w: { layers: [] }, spec: LLAMA, T: 2 };
    const bad = new Float32Array([0, 1, NaN, 2]);
    await assert.rejects(() => Infer.runLayers(S, bad), (e) => e.name === "NonFiniteError");
  });

  console.log(failed ? `\n${failed} failure(s)\n` : "\nall pipeline checks passed\n");
  process.exit(failed ? 1 : 0);
})();
