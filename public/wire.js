// The wire protocol, as pure functions.
//
// Kept out of app.js on purpose: a codec that only exists inside a browser
// event handler cannot be tested, and the failures this protocol can have are
// precisely the ones that are invisible at runtime. A misread offset does not
// throw — it yields a plausible number, and a plausible number in an
// activation is a slightly wrong sentence nobody can attribute to anything.
// So the codec lives here and test_wire.js round-trips it.
//
// Sentinels continue DaisyChain-Web's numbering (it uses -2..-8), so a client
// pointed at the wrong server sees an unknown tag rather than a valid-looking
// message of the wrong kind.
(function (root) {
  "use strict";

  const FRAG = -5,      // fragment of a large message (shared with DaisyChain-Web)
        HELLO = -20,    // capability report
        ASSIGN = -21,   // model identity + which layers you own (no weights)
        READY = -22,    // a stage has fetched its layers and is in the ring
        ACT = -23,      // hidden state moving to the next stage
        TOKEN = -24,    // a token was emitted
        DONE = -25;     // run finished

  // The address of the head's return leg. Not a stage index: the head is both
  // stage 0 and the terminus, so "next = 0" is ambiguous — outbound it means
  // "stage 0, run your blocks", inbound it means "the lap is done". They need
  // distinct addresses or the head re-runs its own blocks and the lap never
  // closes. Negative, so it can never collide with a real stage index.
  const RETURN = -1;

  const enc = new TextEncoder(), dec = new TextDecoder();

  function tagOf(buf) { return new Int32Array(buf, 0, 1)[0]; }

  // ---- hello -----------------------------------------------------------------
  function packHello(capacity, probeHash, backend) {
    const nb = enc.encode(String(backend || "?").slice(0, 32));
    const buf = new ArrayBuffer(16 + nb.length);
    new Int32Array(buf, 0, 1)[0] = HELLO;
    new Float32Array(buf, 4, 1)[0] = capacity;
    new Uint32Array(buf, 8, 1)[0] = probeHash >>> 0;
    new Int32Array(buf, 12, 1)[0] = nb.length;
    new Uint8Array(buf, 16).set(nb);
    return buf;
  }
  function unpackHello(buf) {
    const n = new Int32Array(buf, 12, 1)[0];
    if (n < 0 || 16 + n > buf.byteLength) throw new Error("hello: bad backend length");
    return { capacity: new Float32Array(buf, 4, 1)[0],
             probeHash: new Uint32Array(buf, 8, 1)[0],
             backend: dec.decode(new Uint8Array(buf, 16, n)) };
  }

  // ---- assignment -------------------------------------------------------------
  // Weights do NOT travel between peers. The head sends each device the model's
  // identity and which layers it owns; the device then fetches exactly those
  // tensors from the Hub itself, with its own credentials. So this message is
  // small, and a token never crosses the wire.
  //
  //   [i32 ASSIGN][utf8 JSON]
  //
  // JSON rather than packed ints because the payload is a model spec whose
  // fields differ by architecture. It is validated on arrival — a malformed
  // assignment must fail here, not three layers deep in a GEMM.
  function packAssign(a) {
    const bytes = enc.encode(JSON.stringify(a));
    const buf = new ArrayBuffer(4 + bytes.length);
    new Int32Array(buf, 0, 1)[0] = ASSIGN;
    new Uint8Array(buf, 4).set(bytes);
    return buf;
  }
  function unpackAssign(buf) {
    let a;
    try { a = JSON.parse(dec.decode(new Uint8Array(buf, 4))); }
    catch (e) { throw new Error("assignment: payload is not valid JSON"); }
    if (!a || typeof a.repo !== "string" || !a.repo)
      throw new Error("assignment: no repo id");
    if (!/^[\w.-]+\/[\w.-]+$/.test(a.repo))
      throw new Error(`assignment: "${a.repo}" is not a valid repo id`);
    if (!a.spec || !a.spec.layers || !a.spec.hidden)
      throw new Error("assignment: incomplete model spec");
    if (!Array.isArray(a.plan) || !a.plan.length)
      throw new Error("assignment: no plan");
    // A plan that does not cover every layer exactly once still generates
    // fluent text — with a layer missing. It has to be refused at the door.
    let cover = 0;
    for (const s of a.plan) {
      if (typeof s.lo !== "number" || typeof s.hi !== "number" || s.hi < s.lo || s.lo < 0 || s.hi > a.spec.layers)
        throw new Error("assignment: stage range out of bounds");
      cover += s.hi - s.lo;
    }
    if (cover !== a.spec.layers)
      throw new Error(`assignment: stages cover ${cover} of ${a.spec.layers} layers`);
    if (!a.plan[0].head) throw new Error("assignment: stage 0 must be the head");
    if (typeof a.mine !== "number" || a.mine < 0 || a.mine >= a.plan.length)
      throw new Error("assignment: no valid stage index for this device");
    return a;
  }

  // ---- activation ------------------------------------------------------------
  // [i32 ACT, seq, tokenIdx, nextIndex][u32 actHash, modelHash][f32 hidden]
  // actHash is an integrity check on the payload; modelHash answers a different
  // question — whether this activation belongs to the model I hold a slice of.
  // Two rings running at once on one device is not hypothetical: reload the
  // head with a different checkpoint and the old stages are still out there.
  function packAct(seq, tokenIdx, nextIndex, hidden, hashF32, modelHash) {
    const buf = new ArrayBuffer(24 + hidden.byteLength);
    new Int32Array(buf, 0, 4).set([ACT, seq, tokenIdx, nextIndex]);
    new Uint32Array(buf, 16, 2).set([hashF32(hidden) >>> 0, modelHash >>> 0]);
    new Float32Array(buf, 24).set(hidden);
    return buf;
  }
  function unpackAct(buf, hashF32) {
    if (buf.byteLength < 24) throw new Error("act: truncated header");
    const iv = new Int32Array(buf, 0, 4);
    const [actHash, modelHash] = new Uint32Array(buf, 16, 2);
    const hidden = new Float32Array(buf.slice(24));
    if (hashF32 && (hashF32(hidden) >>> 0) !== actHash) throw new Error("act: payload failed its integrity hash");
    return { seq: iv[1], tokenIdx: iv[2], nextIndex: iv[3], actHash, modelHash, hidden };
  }

  // ---- routing ---------------------------------------------------------------
  // Where an activation goes after this stage, and how to read one that arrives.
  // These are two lines each and they live here, as pure functions, only
  // because getting them wrong is invisible everywhere else: the math is
  // correct, the bytes are correct, and the lap simply never closes. Neither a
  // numeric oracle nor a codec round-trip can see that — it is a property of
  // the ROUTE, so it needs something that can walk one.
  function routeAfter(plan, myIndex) {
    const isLast = myIndex === plan.length - 1;
    const next = isLast ? plan[0] : plan[myIndex + 1];
    return { to: next.id, address: isLast ? RETURN : next.index, isReturn: isLast };
  }
  function classifyAct(nextIndex, myIndex) {
    if (nextIndex === RETURN) return "return";     // the lap is finished
    if (nextIndex === myIndex) return "mine";      // run my blocks, pass it on
    return "other";                                // not addressed to me
  }

  // ---- token / done ----------------------------------------------------------
  function packToken(seq, id, total) {
    const buf = new ArrayBuffer(16);
    new Int32Array(buf).set([TOKEN, seq, id, total]);
    return buf;
  }
  function unpackToken(buf) {
    const iv = new Int32Array(buf, 0, 4);
    return { seq: iv[1], id: iv[2], total: iv[3] };
  }
  function packDone(seq) { return new Int32Array([DONE, seq]).buffer; }

  // ---- ready ------------------------------------------------------------------
  function packReady(stageIndex, ok, note) {
    const bytes = enc.encode(JSON.stringify({ stageIndex, ok: !!ok, note: String(note || "").slice(0, 300) }));
    const buf = new ArrayBuffer(4 + bytes.length);
    new Int32Array(buf, 0, 1)[0] = READY;
    new Uint8Array(buf, 4).set(bytes);
    return buf;
  }
  function unpackReady(buf) {
    try { return JSON.parse(dec.decode(new Uint8Array(buf, 4))); }
    catch (e) { throw new Error("ready: payload is not valid JSON"); }
  }

  const api = { FRAG, HELLO, ASSIGN, READY, ACT, TOKEN, DONE, RETURN, tagOf,
                packHello, unpackHello, packAssign, unpackAssign, packReady, unpackReady,
                packAct, unpackAct, packToken, unpackToken, packDone,
                routeAfter, classifyAct };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Wire = api;
})(typeof self !== "undefined" ? self : this);
