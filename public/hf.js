// Hugging Face Hub client — fetch only the bytes this device needs.
//
// Two rules shape this file.
//
// 1. NOTHING is downloaded whole. config.json and the tokenizer are small and
//    fetched normally; the weights are read with HTTP range requests against
//    the byte offsets in the safetensors header. A device that owns four
//    layers transfers four layers.
//
// 2. The token is asked for ONCE, held in memory, and never written anywhere.
//    Not localStorage, not sessionStorage, not a cookie, not a URL, not the
//    log, and above all never onto the wire — peers each authenticate
//    themselves. A page reload is meant to lose it. See requestToken() in
//    app.js for the prompt; this file only ever receives it as an argument.
(function (root) {
  "use strict";

  let ST;                                  // Safetensors — resolved per environment at the end
  const HUB = "https://huggingface.co";

  // A token that reaches a log or a peer is a leaked credential, so anything
  // that formats an error goes through here first. Cheap, and it means a
  // careless template literal cannot undo rule 2.
  function redact(s, token) {
    if (!token) return s;
    return String(s).split(token).join("hf_***");
  }

  function headers(token, extra) {
    const h = Object.assign({}, extra || {});
    if (token) h["Authorization"] = "Bearer " + token;
    return h;
  }

  function fileUrl(repo, file, revision) {
    return `${HUB}/${repo}/resolve/${encodeURIComponent(revision || "main")}/${file}`;
  }

  // Turn the Hub's status codes into something a person can act on. 401/403 on
  // a public-looking repo almost always means "gated, accept the licence" or
  // "private, needs a token", and those need different actions from the user,
  // so they are not collapsed into one message.
  class HFError extends Error {
    constructor(msg, kind, status) { super(msg); this.kind = kind; this.status = status; }
  }
  function statusError(res, repo, file, hasToken) {
    const where = `${repo}/${file}`;
    if (res.status === 401)
      return new HFError(hasToken
        ? `${where}: the token was rejected (401). Check it is a valid READ token and has not been revoked.`
        : `${where} needs authentication (401).`, "auth", 401);
    if (res.status === 403)
      return new HFError(hasToken
        ? `${where}: this token does not have access (403). If the model is gated, accept its licence on the model page first.`
        : `${where} is gated or private (403).`, "auth", 403);
    if (res.status === 404)
      return new HFError(`${where} not found (404). Check the repo id, and that this model ships .safetensors weights.`, "missing", 404);
    return new HFError(`${where}: HTTP ${res.status}`, "http", res.status);
  }

  async function getJSON(repo, file, revision, token) {
    const res = await fetch(fileUrl(repo, file, revision), { headers: headers(token) });
    if (!res.ok) throw statusError(res, repo, file, !!token);
    try { return await res.json(); }
    catch (e) { throw new HFError(`${repo}/${file} is not valid JSON`, "parse", res.status); }
  }

  // Range read. The Hub redirects to a CDN that honours Range and answers 206;
  // a 200 here means the whole file came back, which for a multi-hundred-MB
  // weights file is exactly the outcome this project exists to avoid — so it
  // is treated as an error rather than silently accepted.
  async function getRange(repo, file, revision, token, start, end) {
    const res = await fetch(fileUrl(repo, file, revision), {
      headers: headers(token, { Range: `bytes=${start}-${end - 1}` }),
    });
    if (res.status === 200)
      throw new HFError(`${repo}/${file}: the server ignored the byte range and returned the whole file — refusing it.`, "norange", 200);
    if (!res.ok && res.status !== 206) throw statusError(res, repo, file, !!token);
    return res.arrayBuffer();
  }

  // ---- what a repo contains ---------------------------------------------------
  // Sharded repos carry model.safetensors.index.json mapping tensor -> shard.
  // Single-file repos just have model.safetensors. Try the index first: its
  // absence is the normal case and a 404 is cheap.
  async function listWeightFiles(repo, revision, token) {
    try {
      const idx = await getJSON(repo, "model.safetensors.index.json", revision, token);
      const files = [...new Set(Object.values(idx.weight_map || {}))];
      if (!files.length) throw new HFError("weight index lists no shards", "parse", 200);
      return { files, weightMap: idx.weight_map };
    } catch (e) {
      if (e.kind === "auth") throw e;                 // don't mask a token problem
      return { files: ["model.safetensors"], weightMap: null };
    }
  }

  // Read just the header of each weight shard: two requests per shard, a few
  // tens of KB, and afterwards we know every tensor's exact byte range without
  // having touched a single weight.
  async function readIndex(repo, revision, token, onProgress) {
    const { files } = await listWeightFiles(repo, revision, token);
    const tensors = new Map();
    for (const f of files) {
      if (onProgress) onProgress(`reading ${f} header…`);
      const head = await getRange(repo, f, revision, token, 0, 8);
      const dv = new DataView(head);
      const lo = dv.getUint32(0, true), hi = dv.getUint32(4, true);
      if (hi !== 0) throw new HFError(`${f}: implausible header length — not a safetensors file`, "parse", 200);
      const full = await getRange(repo, f, revision, token, 0, 8 + lo);
      const parsed = ST.parseHeader(full);
      for (const [name, t] of parsed.tensors) tensors.set(name, Object.assign({ file: f }, t));
    }
    return { files, tensors };
  }

  // Fetch a specific set of tensors as f32, coalescing adjacent byte ranges.
  // This is the only function that moves real weight data.
  async function fetchTensors(repo, revision, token, tensors, onProgress) {
    const out = new Map();
    const byFile = new Map();
    for (const t of tensors) {
      if (!byFile.has(t.file)) byFile.set(t.file, []);
      byFile.get(t.file).push(t);
    }
    let done = 0;
    const total = tensors.reduce((a, t) => a + t.bytes, 0);
    for (const [file, list] of byFile) {
      for (const run of ST.coalesce(list)) {
        const buf = await getRange(repo, file, revision, token, run.start, run.end);
        for (const t of run.tensors) {
          const off = t.start - run.start;
          out.set(t.name, ST.toF32(t.dtype, new Uint8Array(buf, off, t.bytes)));
          done += t.bytes;
          if (onProgress) onProgress(`${(done / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);
        }
      }
    }
    return out;
  }

  // A cheap probe so the UI can ask for a token only when one is actually
  // needed, instead of demanding credentials up front for public models.
  async function needsAuth(repo, revision) {
    try {
      const res = await fetch(fileUrl(repo, "config.json", revision), { method: "HEAD" });
      return res.status === 401 || res.status === 403;
    } catch (e) { return false; }
  }

  const api = { HUB, HFError, fileUrl, getJSON, getRange, listWeightFiles, readIndex,
                fetchTensors, needsAuth, redact };
  if (typeof module !== "undefined" && module.exports) { ST = require("./safetensors.js"); module.exports = api; }
  else { ST = root.Safetensors; root.HF = api; }
})(typeof self !== "undefined" ? self : this);
