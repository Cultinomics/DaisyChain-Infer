// safetensors, read a piece at a time.
//
// The format is what makes this project's memory pooling honest. A file is:
//
//   [u64 headerLen][JSON header][ tensor bytes, back to back ]
//
// and the header gives every tensor's exact byte range. So a device does not
// need the model — it needs its own layers, and it can ask the CDN for exactly
// those bytes. Nothing else is ever downloaded, decoded, or held.
//
// That is the difference between "we split the model across devices" and
// "one device downloads the model and hands out pieces". Only the first one
// actually lets a group run something none of them could hold.
(function (root) {
  "use strict";

  // ---- dtype decoding ---------------------------------------------------------
  // Everything becomes f32, because that is what the verified units quantize
  // from. F16 and BF16 are the common storage formats and both widen exactly —
  // no rounding, no device-dependent behaviour, so a CPU device and a GPU
  // device still agree bit-for-bit after loading.
  const BYTES = { F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 };

  // BF16 is the top 16 bits of an f32 — widening is a shift, exactly.
  function bf16ToF32(u16, out) {
    const u32 = new Uint32Array(out.buffer, out.byteOffset, out.length);
    for (let i = 0; i < u16.length; i++) u32[i] = u16[i] << 16;
    return out;
  }
  // F16 -> F32 by hand rather than via Float16Array, which is not everywhere
  // yet. Subnormals and inf/nan are handled explicitly; every f16 has an exact
  // f32 representation, so this is a widening, not a conversion.
  function f16ToF32(u16, out) {
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = (h & 0x8000) >> 15, e = (h & 0x7C00) >> 10, f = h & 0x03FF;
      let v;
      if (e === 0) v = f === 0 ? 0 : Math.pow(2, -14) * (f / 1024);
      else if (e === 0x1F) v = f === 0 ? Infinity : NaN;
      else v = Math.pow(2, e - 15) * (1 + f / 1024);
      out[i] = s ? -v : v;
    }
    return out;
  }

  function toF32(dtype, bytes) {
    const n = bytes.byteLength / BYTES[dtype];
    if (dtype === "F32") {
      // may be unaligned inside a range response — copy rather than view
      const out = new Float32Array(n);
      new Uint8Array(out.buffer).set(bytes);
      return out;
    }
    const out = new Float32Array(n);
    if (dtype === "F16" || dtype === "BF16") {
      const u16 = new Uint16Array(n);
      new Uint8Array(u16.buffer).set(bytes);
      return dtype === "BF16" ? bf16ToF32(u16, out) : f16ToF32(u16, out);
    }
    if (dtype === "F64") {
      const f64 = new Float64Array(n);
      new Uint8Array(f64.buffer).set(bytes);
      for (let i = 0; i < n; i++) out[i] = f64[i];
      return out;
    }
    throw new Error(`unsupported tensor dtype ${dtype} — weights must be F32, F16, BF16 or F64`);
  }

  // ---- header ----------------------------------------------------------------
  // The first 8 bytes are a u64 length. It is read as two u32s because a JS
  // number cannot hold a u64 — but a header that genuinely needed the high word
  // would be gigabytes, so a nonzero high word means the file is not what it
  // claims and is refused rather than truncated into something plausible.
  function parseHeader(buf) {
    const dv = new DataView(buf);
    const lo = dv.getUint32(0, true), hi = dv.getUint32(4, true);
    if (hi !== 0) throw new Error("safetensors header length is implausibly large — not a safetensors file");
    if (lo <= 0 || lo > buf.byteLength - 8) throw new Error("safetensors header is truncated");
    let json;
    try { json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, lo))); }
    catch (e) { throw new Error("safetensors header is not valid JSON — not a safetensors file"); }
    const dataStart = 8 + lo;
    const tensors = new Map();
    for (const [name, t] of Object.entries(json)) {
      if (name === "__metadata__") continue;
      if (!t || !t.dtype || !Array.isArray(t.shape) || !Array.isArray(t.data_offsets))
        throw new Error(`safetensors header entry "${name}" is malformed`);
      const [s, e] = t.data_offsets;
      const elems = t.shape.reduce((a, b) => a * b, 1);
      const want = elems * (BYTES[t.dtype] || 0);
      if (BYTES[t.dtype] && e - s !== want)
        throw new Error(`tensor "${name}": header claims ${e - s} bytes for a ${t.shape.join("x")} ${t.dtype} (expected ${want})`);
      tensors.set(name, { name, dtype: t.dtype, shape: t.shape, elems,
                          start: dataStart + s, end: dataStart + e, bytes: e - s });
    }
    return { tensors, dataStart, metadata: json.__metadata__ || {} };
  }

  // How much a set of tensors will cost to hold, before fetching any of it.
  // Reported in the UI so a device can see whether its slice fits BEFORE it
  // spends the bandwidth finding out.
  function f32Bytes(tensors) {
    let n = 0;
    for (const t of tensors) n += t.elems * 4;
    return n;
  }

  // ---- coalescing -------------------------------------------------------------
  // A stage's tensors are contiguous in the file far more often than not
  // (safetensors writes them in the order they were registered, which follows
  // layer order). Merging adjacent ranges turns ~9 requests per layer into
  // roughly one per stage. Gaps below `slack` are downloaded and discarded
  // because one extra request costs more than a few unused kilobytes.
  function coalesce(tensors, slack) {
    const gap = slack == null ? 1 << 20 : slack;
    const sorted = [...tensors].sort((a, b) => a.start - b.start);
    const runs = [];
    for (const t of sorted) {
      const last = runs[runs.length - 1];
      if (last && t.start - last.end <= gap) { last.end = Math.max(last.end, t.end); last.tensors.push(t); }
      else runs.push({ start: t.start, end: t.end, tensors: [t] });
    }
    return runs;
  }

  const api = { BYTES, parseHeader, toF32, coalesce, f32Bytes, f16ToF32, bf16ToF32 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Safetensors = api;
})(typeof self !== "undefined" ? self : this);
