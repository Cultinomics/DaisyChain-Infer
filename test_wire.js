// Protocol round-trips, and the malformed messages the protocol must refuse.
//
// The reason this file exists separately from test_pipeline.js: the pipeline
// test proves the MATH survives being split. It cannot see a protocol bug,
// because it never encodes anything — it hands Float32Arrays between stages
// directly. Every real failure of a distributed system lives in the gap those
// two tests leave for each other, which is exactly what DaisyChain-Web's own
// self-corpus writeup concluded after a stalled roster gradient turned out to
// be a protocol bug that no data oracle could fire on.
//
//   node test_wire.js
const assert = require("assert");
const Wire = require("./public/wire.js");
const Shard = require("./public/shard.js");

const SPEC = { family: "llama", layers: 6, hidden: 64, heads: 4, kvHeads: 2, vocab: 128 };
const PLAN = [{ id: "p1", index: 0, lo: 0, hi: 2, head: true },
              { id: "p7", index: 1, lo: 2, hi: 4, head: false },
              { id: "p3", index: 2, lo: 4, hi: 6, head: false }];
function assign(over) { return Object.assign({ repo: "owner/model", revision: "main", spec: SPEC, plan: PLAN, mine: 1, fingerprint: 123 }, over || {}); }

function bitsEqual(a, b) {
  if (a.length !== b.length) return false;
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
}

let failed = 0;
function t(name, fn) {
  try { fn(); console.log(`  ok    ${name}`); }
  catch (e) { failed++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}

console.log("\nDaisyChain-Infer — wire protocol\n");

t("hello round-trips, including the backend string", () => {
  const b = Wire.packHello(1234.5, 0xDEADBEEF, "webgpu");
  assert.strictEqual(Wire.tagOf(b), Wire.HELLO);
  const h = Wire.unpackHello(b);
  assert.strictEqual(Math.fround(1234.5), h.capacity);
  assert.strictEqual(h.probeHash, 0xDEADBEEF);
  assert.strictEqual(h.backend, "webgpu");
});

t("assignment round-trips the model identity and the plan", () => {
  const a = Wire.unpackAssign(Wire.packAssign(assign()));
  assert.strictEqual(a.repo, "owner/model");
  assert.strictEqual(a.revision, "main");
  assert.strictEqual(a.mine, 1);
  assert.strictEqual(a.spec.layers, 6);
  assert.deepStrictEqual(a.plan.map(s2 => [s2.lo, s2.hi]), [[0, 2], [2, 4], [4, 6]]);
});

// An assignment carries no weights by design — each device fetches its own
// layers from the Hub. This asserts the message stays small no matter how big
// the model is, which is what keeps a token off the wire and a peer from ever
// shipping weights to another peer.
t("an assignment carries no weight data", () => {
  const buf = Wire.packAssign(assign({ spec: Object.assign({}, SPEC, { hidden: 4096, layers: 80, vocab: 128000 }) }));
  assert.ok(buf.byteLength < 2048, `assignment is ${buf.byteLength} bytes — it must never carry weights`);
});

// The failure this rejects produces a plausible answer rather than an error:
// a plan missing a layer still generates fluent text.
t("a plan that does not cover every layer is refused", () => {
  const gap = [{ id: "p1", lo: 0, hi: 2, head: true }, { id: "p2", lo: 3, hi: 6, head: false }];
  assert.throws(() => Wire.unpackAssign(Wire.packAssign(assign({ plan: gap }))), /cover 5 of 6/);
  const over = [{ id: "p1", lo: 0, hi: 4, head: true }, { id: "p2", lo: 2, hi: 6, head: false }];
  assert.throws(() => Wire.unpackAssign(Wire.packAssign(assign({ plan: over }))), /cover 8 of 6/);
  const oob = [{ id: "p1", lo: 0, hi: 9, head: true }];
  assert.throws(() => Wire.unpackAssign(Wire.packAssign(assign({ plan: oob }))), /out of bounds/);
});

t("an assignment whose stage 0 is not the head is refused", () => {
  const bad = [{ id: "p1", lo: 0, hi: 3, head: false }, { id: "p2", lo: 3, hi: 6, head: true }];
  assert.throws(() => Wire.unpackAssign(Wire.packAssign(assign({ plan: bad }))), /stage 0 must be the head/);
});

// A repo id is interpolated straight into a Hub URL, so it is validated rather
// than trusted: a peer should not be able to point this device at an arbitrary
// path by sending a crafted assignment.
t("a malformed repo id is refused", () => {
  for (const repo of ["", "nope", "../../etc/passwd", "owner/name/extra", "owner/na me"])
    assert.throws(() => Wire.unpackAssign(Wire.packAssign(assign({ repo }))), /repo id/, `accepted "${repo}"`);
});

t("an assignment with no valid stage index for this device is refused", () => {
  assert.throws(() => Wire.unpackAssign(Wire.packAssign(assign({ mine: 9 }))), /stage index/);
  assert.throws(() => Wire.unpackAssign(Wire.packAssign(assign({ mine: -1 }))), /stage index/);
});

t("ready round-trips", () => {
  const r = Wire.unpackReady(Wire.packReady(2, true, "loaded"));
  assert.strictEqual(r.stageIndex, 2);
  assert.strictEqual(r.ok, true);
  const bad = Wire.unpackReady(Wire.packReady(1, false, "out of memory"));
  assert.strictEqual(bad.ok, false);
  assert.strictEqual(bad.note, "out of memory");
});

t("activation round-trips and carries both hashes", () => {
  const hidden = new Float32Array(16 * 32);
  for (let i = 0; i < hidden.length; i++) hidden[i] = Math.cos(i) * 0.5;
  const buf = Wire.packAct(7, 3, 2, hidden, Shard.hashF32, 0x5150);
  const a = Wire.unpackAct(buf, Shard.hashF32);
  assert.strictEqual(a.seq, 7);
  assert.strictEqual(a.tokenIdx, 3);
  assert.strictEqual(a.nextIndex, 2);
  assert.strictEqual(a.modelHash, 0x5150);
  assert.ok(bitsEqual(a.hidden, hidden), "hidden state changed on the wire");
});

// Corruption in an activation is the quietest failure in the whole system:
// the next stage happily consumes any float array of the right length.
t("a corrupted activation is caught by its hash", () => {
  const hidden = new Float32Array(64).fill(1.5);
  const buf = Wire.packAct(1, 0, 1, hidden, Shard.hashF32, 9);
  new Float32Array(buf, 24)[10] = 1.5000001;              // one ulp, deep in the payload
  assert.throws(() => Wire.unpackAct(buf, Shard.hashF32), /integrity hash/);
});

t("-0 in an activation survives, and is distinguished from +0", () => {
  // The parent project learned this the hard way: `!==` says -0 === 0, but the
  // hash sees the bits. An activation carrying -0 must arrive as -0, and a
  // flip to +0 must not slip past the integrity check.
  const hidden = new Float32Array(8);
  hidden[3] = -0;
  const buf = Wire.packAct(1, 0, 1, hidden, Shard.hashF32, 9);
  const a = Wire.unpackAct(buf, Shard.hashF32);
  assert.ok(Object.is(a.hidden[3], -0), "-0 did not survive the wire");
  new Float32Array(buf, 24)[3] = 0;
  assert.throws(() => Wire.unpackAct(buf, Shard.hashF32), /integrity hash/);
});

t("token and done round-trip", () => {
  const tk = Wire.unpackToken(Wire.packToken(4, 1337, 42));
  assert.deepStrictEqual([tk.seq, tk.id, tk.total], [4, 1337, 42]);
  assert.strictEqual(Wire.tagOf(Wire.packDone(9)), Wire.DONE);
});

// ---- routing ---------------------------------------------------------------
// These exist because of a real bug, and they are the shape of test that would
// have caught it. The head is both stage 0 and the ring's terminus, so the
// returning activation was addressed to index 0 and every stage read that as
// "stage 0, run your blocks" — the head re-ran its own layers and forwarded
// again, and the lap never closed. Every number involved was correct. Every
// message round-tripped. The pipeline test could not see it because it calls
// the stages in order itself, and the codec test could not see it because the
// bytes were fine. The defect was in the ROUTE, so the check has to walk one.
function walkLap(plan) {
  // Simulate one lap: start at stage 0, follow routeAfter/classifyAct, and
  // record who runs. Returns the sequence of stage indices that computed.
  const ran = [];
  let addr = 0, guard = 0;
  for (;;) {
    if (++guard > 100) throw new Error("lap did not terminate — the ring is cycling");
    let actor = null;
    for (const s of plan) if (Wire.classifyAct(addr, s.index) === "mine") {
      if (actor !== null) throw new Error(`two stages both claim address ${addr}`);
      actor = s.index;
    }
    if (actor === null) {
      if (Wire.classifyAct(addr, 0) !== "return") throw new Error(`address ${addr} is claimed by nobody`);
      return ran;                                   // lap closed at the head
    }
    ran.push(actor);
    addr = Wire.routeAfter(plan, actor).address;
  }
}

t("a lap visits every stage exactly once and terminates at the head", () => {
  for (const n of [1, 2, 3, 5, 8]) {
    const plan = Array.from({ length: n }, (_, i) => ({ id: "p" + i, index: i, lo: i, hi: i + 1, head: i === 0 }));
    assert.deepStrictEqual(walkLap(plan), plan.map(s => s.index),
      `a ${n}-stage ring did not visit each stage exactly once in order`);
  }
});

t("the return leg is addressed differently from stage 0", () => {
  const plan = [{ id: "p0", index: 0 }, { id: "p1", index: 1 }];
  const last = Wire.routeAfter(plan, 1);
  assert.strictEqual(last.to, "p0", "the last stage must hand back to the head");
  assert.ok(last.isReturn, "the last hop must be flagged as the return leg");
  assert.notStrictEqual(last.address, 0,
    "the return leg is addressed as stage 0 — the head will re-run its own blocks and the lap will never close");
  assert.strictEqual(Wire.classifyAct(last.address, 0), "return");
  assert.strictEqual(Wire.classifyAct(0, 0), "mine");     // outbound, same number, different meaning
});

t("a single-stage ring returns to itself immediately", () => {
  const solo = [{ id: "p0", index: 0 }];
  const hop = Wire.routeAfter(solo, 0);
  assert.ok(hop.isReturn && hop.to === "p0");
  assert.deepStrictEqual(walkLap(solo), [0]);
});

t("every sentinel is distinct and cannot collide with a step number", () => {
  const tags = [Wire.FRAG, Wire.HELLO, Wire.ASSIGN, Wire.READY, Wire.ACT, Wire.TOKEN, Wire.DONE];
  assert.strictEqual(new Set(tags).size, tags.length, "two sentinels share a value");
  for (const t2 of tags) assert.ok(t2 < 0, `sentinel ${t2} is not negative — it could be mistaken for data`);
  // DaisyChain-Web uses -2..-8; ours start at -20 so a misdirected client sees
  // an unknown tag rather than a valid message of the wrong kind.
  for (const t2 of tags) assert.ok(t2 === Wire.FRAG || t2 <= -20, `sentinel ${t2} overlaps DaisyChain-Web's range`);
});

console.log(failed ? `\n${failed} failure(s)\n` : "\nall wire checks passed\n");
process.exit(failed ? 1 : 0);
