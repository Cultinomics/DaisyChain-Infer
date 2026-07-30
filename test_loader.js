// The loader: safetensors parsing, dtype widening, range coalescing, and BPE.
//
// This is the layer where "load any model off HuggingFace" is either true or
// quietly false. Every failure here is silent by nature — a misread offset
// gives plausible floats, a wrong dtype gives plausible floats, a
// half-implemented tokenizer gives plausible text. So the checks are exact,
// and several are built by CONSTRUCTING a file and reading it back rather than
// by trusting a fixture.
//
//   node test_loader.js
const assert = require("assert");
const ST = require("./public/safetensors.js");
const Tok = require("./public/tokenizer.js");
const Arch = require("./public/arch.js");

let failed = 0;
function t(name, fn) {
  try { fn(); console.log(`  ok    ${name}`); }
  catch (e) { failed++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}

console.log("\nDaisyChain-Infer — loader\n");

// Build a real safetensors buffer so the parser is tested against the format,
// not against our own idea of it.
function buildFile(entries) {
  let offset = 0;
  const header = {};
  const chunks = [];
  for (const [name, dtype, shape, bytes] of entries) {
    header[name] = { dtype, shape, data_offsets: [offset, offset + bytes.length] };
    chunks.push(bytes);
    offset += bytes.length;
  }
  const hj = Buffer.from(JSON.stringify(header), "utf8");
  const out = Buffer.alloc(8 + hj.length + offset);
  out.writeUInt32LE(hj.length, 0); out.writeUInt32LE(0, 4);
  hj.copy(out, 8);
  let o = 8 + hj.length;
  for (const c of chunks) { Buffer.from(c).copy(out, o); o += c.length; }
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.length);
}

t("parses a header and locates every tensor", () => {
  const a = Buffer.from(new Float32Array([1, 2, 3, 4]).buffer);
  const b = Buffer.from(new Float32Array([5, 6]).buffer);
  const buf = buildFile([["w.a", "F32", [2, 2], a], ["w.b", "F32", [2], b]]);
  const { tensors } = ST.parseHeader(buf);
  assert.strictEqual(tensors.size, 2);
  const ta = tensors.get("w.a");
  assert.deepStrictEqual(ta.shape, [2, 2]);
  assert.strictEqual(ta.elems, 4);
  assert.deepStrictEqual([...ST.toF32("F32", new Uint8Array(buf, ta.start, ta.bytes))], [1, 2, 3, 4]);
  const tb = tensors.get("w.b");
  assert.deepStrictEqual([...ST.toF32("F32", new Uint8Array(buf, tb.start, tb.bytes))], [5, 6]);
});

// A shape that disagrees with the byte range means every subsequent tensor
// would be read at the wrong offset — plausible floats all the way down.
t("a header whose shape contradicts its byte range is refused", () => {
  const a = Buffer.from(new Float32Array([1, 2, 3, 4]).buffer);
  const buf = buildFile([["w", "F32", [9, 9], a]]);
  assert.throws(() => ST.parseHeader(buf), /expected/);
});

t("junk in place of a header is refused, not guessed at", () => {
  const junk = Buffer.alloc(64); junk.writeUInt32LE(16, 0);
  assert.throws(() => ST.parseHeader(junk.buffer.slice(0, 64)), /not valid JSON|not a safetensors/);
  const big = Buffer.alloc(16); big.writeUInt32LE(0, 0); big.writeUInt32LE(7, 4);
  assert.throws(() => ST.parseHeader(big.buffer.slice(0, 16)), /implausibly large/);
});

// BF16 is the top 16 bits of an f32, so widening must be EXACT — not close.
t("BF16 widens exactly (it is the high half of an f32)", () => {
  const vals = [1, -1, 0.5, -0.5, 2, 100, -0.0078125, 0];
  const u16 = new Uint16Array(vals.length);
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  for (let i = 0; i < vals.length; i++) { f[0] = vals[i]; u16[i] = u[0] >>> 16; }
  const out = ST.bf16ToF32(u16, new Float32Array(vals.length));
  for (let i = 0; i < vals.length; i++)
    assert.strictEqual(out[i], vals[i], `bf16 round-trip changed ${vals[i]} to ${out[i]}`);
});

t("F16 widens exactly, including subnormals and signed zero", () => {
  // every f16 has an exact f32 value, so this is a widening with no error
  const cases = [[0x3C00, 1], [0xBC00, -1], [0x3800, 0.5], [0x0000, 0], [0x8000, -0],
                 [0x0001, Math.pow(2, -24)], [0x7BFF, 65504]];
  const u16 = Uint16Array.from(cases.map(c => c[0]));
  const out = ST.f16ToF32(u16, new Float32Array(cases.length));
  cases.forEach(([, want], i) =>
    assert.ok(Object.is(out[i], want), `f16 ${i}: got ${out[i]}, want ${want}`));
});

// Coalescing is a bandwidth optimisation, and a wrong one would hand a tensor
// the wrong bytes. It must merge only what is genuinely adjacent, and the
// per-tensor offsets must still land correctly inside a merged run.
t("range coalescing merges neighbours and keeps offsets correct", () => {
  const mk = (start, end) => ({ start, end, bytes: end - start, elems: (end - start) / 4, dtype: "F32", name: `t${start}` });
  const runs = ST.coalesce([mk(0, 100), mk(100, 200), mk(10_000_000, 10_000_100)], 1024);
  assert.strictEqual(runs.length, 2, "adjacent ranges were not merged, or a distant one was");
  assert.deepStrictEqual([runs[0].start, runs[0].end], [0, 200]);
  assert.strictEqual(runs[0].tensors.length, 2);
  assert.deepStrictEqual([runs[1].start, runs[1].end], [10_000_000, 10_000_100]);
  // a tensor's data must be recoverable from its offset within the merged run
  const t2 = runs[0].tensors[1];
  assert.strictEqual(t2.start - runs[0].start, 100);
});

t("stage byte cost is computed from the header alone", () => {
  const spec = Arch.fromConfig({ model_type: "llama", hidden_size: 64, num_hidden_layers: 4,
    num_attention_heads: 4, num_key_value_heads: 4, intermediate_size: 128, vocab_size: 100 });
  const names = new Map();
  const add = (n, elems) => names.set(n, { name: n, elems, bytes: elems * 2, dtype: "BF16", shape: [elems] });
  const h = Arch.headTensors(spec);
  add(h.emb, 100 * 64); add(h.nrmF, 64);
  for (let l = 0; l < 4; l++) {
    const lt = Arch.layerTensors(spec, l);
    add(lt.nrm1, 64); add(lt.nrm2, 64);
    add(lt.Wq, 64 * 64); add(lt.Wk, 64 * 64); add(lt.Wv, 64 * 64); add(lt.Wo, 64 * 64);
    add(lt.Wgate, 128 * 64); add(lt.Wup, 128 * 64); add(lt.Wdown, 64 * 128);
  }
  const Shard = require("./public/shard.js");
  const mid = Shard.stageBytes(spec, names, { lo: 1, hi: 2, head: false }, Arch);
  const head = Shard.stageBytes(spec, names, { lo: 0, hi: 1, head: true }, Arch);
  assert.strictEqual(mid, (4 * 64 * 64 + 3 * 128 * 64 + 2 * 64) * 4, "middle stage cost is wrong");
  assert.ok(head > mid, "the head must cost more — it holds the embedding table");
});

// ---- tokenizer --------------------------------------------------------------
// A tiny byte-level BPE built by hand, so encode/decode are checked against a
// known merge table rather than against themselves.
const TOKJSON = (() => {
  const vocab = {};
  let id = 0;
  // single byte-level characters for the printable ASCII we use
  for (const ch of "ĠabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!") vocab[ch] = id++;
  vocab["ab"] = id++; vocab["abc"] = id++; vocab["Ġth"] = id++; vocab["Ġthe"] = id++;
  const merges = ["a b", "ab c", "Ġ t h", "Ġth e"].map(m => m.replace(/^(\S+) (\S+)$/, "$1 $2"));
  return { model: { type: "BPE", vocab, merges: ["a b", "ab c", "Ġt h", "Ġth e"] },
           added_tokens: [{ id: id++, content: "<|end|>" }] };
})();

t("byte-level BPE round-trips text exactly", () => {
  const tk = Tok.build(TOKJSON);
  for (const s of ["abc", "abc abc", "the", " the", "hello world", "a,b!c."]) {
    const ids = Tok.encode(tk, s);
    assert.strictEqual(Tok.decode(tk, ids), s, `"${s}" did not round-trip (got "${Tok.decode(tk, ids)}")`);
  }
});

t("BPE actually applies its merges", () => {
  const tk = Tok.build(TOKJSON);
  // "abc" must become the single merged token, not three characters
  assert.deepStrictEqual([...Tok.encode(tk, "abc")], [tk.vocab["abc"]], "merges were not applied");
  assert.deepStrictEqual([...Tok.encode(tk, "ab")], [tk.vocab["ab"]]);
});

t("added/special tokens stay whole and are hidden on decode", () => {
  const tk = Tok.build(TOKJSON);
  const ids = [...Tok.encode(tk, "abc<|end|>abc")];
  assert.ok(ids.includes(tk.added.get("<|end|>")), "the special token was split up");
  assert.strictEqual(Tok.decode(tk, ids), "abcabc", "special tokens should not appear in output text");
  assert.ok(Tok.decode(tk, ids, true).includes("<|end|>"), "keepSpecial should show them");
});

t("a non-BPE tokenizer is refused rather than approximated", () => {
  assert.throws(() => Tok.build({ model: { type: "Unigram", vocab: {} } }), /not supported/);
  assert.throws(() => Tok.build({ model: { type: "WordPiece", vocab: {} } }), /not supported/);
});

t("the byte table covers all 256 bytes and is a bijection", () => {
  const { b2u, u2b } = Tok.byteTables();
  assert.strictEqual(b2u.size, 256, "byte->unicode table is incomplete");
  assert.strictEqual(u2b.size, 256, "unicode->byte table is incomplete");
  for (let b = 0; b < 256; b++) assert.strictEqual(u2b.get(b2u.get(b)), b, `byte ${b} does not round-trip`);
});

console.log(failed ? `\n${failed} failure(s)\n` : "\nall loader checks passed\n");
process.exit(failed ? 1 : 0);
