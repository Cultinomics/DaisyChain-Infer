// DaisyChain-Infer client: connect P2P, take a slice of a model straight from
// the Hub, and run one token loop around the ring.
//
// The mesh (signaling, WebRTC, fragmentation, rooms, reconnect) is
// DaisyChain-Web's, because those are the parts already proven against real
// phones and real NATs. Everything from "the model" down is new.
"use strict";

const STUN = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "turn:openrelay.metered.ca:80", username: "openrelayproject", credential: "openrelayproject" },
  { urls: "turn:openrelay.metered.ca:443", username: "openrelayproject", credential: "openrelayproject" },
  { urls: "turns:openrelay.metered.ca:443?transport=tcp", username: "openrelayproject", credential: "openrelayproject" },
];
let rtcConfig = { iceServers: STUN };

const ui = {};
for (const id of ["status", "backend", "me", "peers", "log", "requests", "roomInfo", "roomCode",
                  "copyLink", "model", "tokenizer", "plan", "myShard", "prompt", "genBtn",
                  "stopBtn", "out", "tps", "tokcount", "hops", "temp", "vtemp", "ntok", "vntok",
                  "seed", "verify", "verifyRow", "ringState", "repo", "revision", "loadRepo",
                  "modelState", "ctxLen", "vctxLen", "tokenState", "clearToken",
                  "tokModal", "tokInput", "tokOk", "tokCancel", "tokWhy",
                  "tokPasteRow", "tokSignIn", "tokSignInBtn",
                  "lobby", "createRoom", "joinRoom", "joinCode"])
  ui[id] = document.getElementById(id);

function log(m) { ui.log.textContent = `${new Date().toLocaleTimeString()}  ${redact(m)}\n` + ui.log.textContent; }
function setStatus(s) { ui.status.textContent = s; }

// ---- the token --------------------------------------------------------------
// Held in a closure variable and nowhere else. Not localStorage, not
// sessionStorage, not a cookie, not the URL, not the log, and never on the
// wire — each device authenticates itself, so a peer never needs anyone
// else's credentials. Reloading the page is meant to lose it.
let hfToken = null, tokenSource = "";
let oauthAvailable = false;
function redact(s) { return hfToken ? String(s).split(hfToken).join("hf_***") : String(s); }
function haveToken() { return !!hfToken; }
function setToken(t, source) {
  hfToken = t || null;
  tokenSource = t ? (source || "entered") : "";
  ui.tokenState.textContent = hfToken
    ? `${tokenSource} · held in memory for this tab only`
    : (oauthAvailable ? "not signed in" : "none");
  ui.clearToken.style.display = hfToken ? "" : "none";
}
function clearToken() { setToken(null); log("token cleared from memory"); }

// A hosted deployment hands the OAuth token back in the URL *fragment*, which
// browsers never send to a server. Take it, then strip it from the address bar
// and from history immediately — a credential sitting in a visible URL is one
// screenshot away from being shared.
function adoptTokenFromFragment() {
  if (!location.hash || location.hash.length < 2) return;
  const h = new URLSearchParams(location.hash.slice(1));
  const t = h.get("hf"), err = h.get("oauth_error");
  if (t || err) history.replaceState(null, "", location.pathname + location.search);
  if (err) { log(`sign-in failed: ${err}`); return; }
  if (t) { setToken(t, "signed in with Hugging Face"); log("signed in with Hugging Face — token held in memory for this tab only"); }
}

// One-time prompt. Resolves to a token string, or null if the user declines.
// Deliberately a modal rather than a stored setting: the moment a credential
// becomes ambient, it becomes something you forget you granted.
// Where OAuth is available — a deployment the user does not control — the
// paste box is not offered at all. Asking someone to type a personal access
// token into a page served by a third party is a bad pattern even when the
// code is honest, because the user cannot verify that it is. Signing in hands
// over a scoped, expiring token instead, and it is the only path shown there.
let tokenPrompt = null;
function requestToken(why) {
  if (tokenPrompt) return tokenPrompt;
  ui.tokWhy.textContent = why || "This model needs a Hugging Face token to download.";
  ui.tokPasteRow.style.display = oauthAvailable ? "none" : "";
  ui.tokSignIn.style.display = oauthAvailable ? "" : "none";
  ui.tokOk.style.display = oauthAvailable ? "none" : "";      // nothing to submit
  ui.tokInput.value = "";
  ui.tokModal.style.display = "flex";
  if (!oauthAvailable) ui.tokInput.focus();
  tokenPrompt = new Promise((resolve) => {
    const done = (val) => {
      ui.tokModal.style.display = "none";
      ui.tokInput.value = "";                    // do not leave it in the DOM
      ui.tokOk.onclick = null; ui.tokCancel.onclick = null;
      ui.tokInput.onkeydown = null; ui.tokSignInBtn.onclick = null;
      tokenPrompt = null;
      resolve(val);
    };
    ui.tokSignInBtn.onclick = () => {
      // full-page redirect: we come back with the token in the fragment
      log("redirecting to Hugging Face to sign in…");
      location.href = "auth/login";
    };
    ui.tokOk.onclick = () => {
      const v = ui.tokInput.value.trim();
      if (!v) return done(null);
      setToken(v, "entered by hand");
      log("token accepted — kept in memory for this tab only, never sent to peers");
      done(v);
    };
    ui.tokCancel.onclick = () => { log("token request declined"); done(null); };
    ui.tokInput.onkeydown = (e) => { if (e.key === "Enter") ui.tokOk.onclick(); if (e.key === "Escape") ui.tokCancel.onclick(); };
  });
  return tokenPrompt;
}

// Run an HF call, and if it fails for want of credentials ask once, then retry.
async function withAuth(fn, what) {
  try { return await fn(hfToken); }
  catch (e) {
    if (e && e.kind === "auth") {
      const gated = e.status === 403 ? " It is gated or private" : "";
      const t = await requestToken(oauthAvailable
        ? `${what} needs Hugging Face access.${gated}${gated ? " — for a gated model, accept its licence on the model page first." : ""}`
        : `${what} needs a Hugging Face token.${gated}. Create a READ token at huggingface.co/settings/tokens.`);
      if (!t) throw new Error(`${what}: cancelled — no token provided`);
      return fn(t);
    }
    throw e;
  }
}

// ---- identity ---------------------------------------------------------------
const ADJ = ["Frosted", "Silver", "Pale", "Winter", "Hollow", "Quiet", "Drifting", "Little",
             "Glacier", "Misty", "Northern", "Still", "Brave", "Snowlit", "Amber"];
const NOUN = ["Fox", "Hare", "Owl", "Elk", "Marten", "Sparrow", "Otter", "Deer",
              "Ptarmigan", "Pine", "Birch", "Wren", "Fawn", "Moth", "Lynx"];
const deviceName = ADJ[Math.floor(Math.random() * ADJ.length)] + " " +
                   NOUN[Math.floor(Math.random() * NOUN.length)];

let myId = null, compute = null, ws = null, L = null, wasDenied = false;
const pcs = new Map(), chans = new Map(), names = new Map();
const caps = new Map();

// ---- model state ------------------------------------------------------------
let repoInfo = null;      // {repo, revision, spec, tensors, fingerprint} — head only
let spec = null, plan = null, myStage = null, stageObj = null;
let tok = null;           // tokenizer, loaded from the same repo
let modelHash = 0;
let running = false, abortRun = false;
let probeHash = 0, auditFailure = null;
const readyStages = new Set();

function nmeOf(id) { return names.get(id) || id; }
function room() { return new URLSearchParams(location.search).get("room"); }
// ?relay=1 skips WebRTC entirely and routes through the server. It exists as a
// diagnostic: if a ring works with it and not without it, the problem is NAT
// traversal, not this code.
const forceRelay = new URLSearchParams(location.search).get("relay") === "1";
function updatePeers() {
  if (!names.size) { ui.peers.textContent = "(none yet — you can still run solo)"; return; }
  ui.peers.textContent = [...names.entries()].map(([id, n]) =>
    `${n} ${chans.has(id) ? (relayed.has(id) ? "⇄ via server" : "✓") : "(connecting…)"}`).join(", ");
}
function ctxLen() { return +ui.ctxLen.value; }

// ---- signaling + WebRTC -----------------------------------------------------
function connectSignaling() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const params = new URLSearchParams();
  params.set("name", deviceName);
  if (room()) params.set("room", room());
  ws = new WebSocket(`${proto}://${location.host}/?${params}`);
  ws.onopen = () => setStatus(room() ? `connected — private room "${room()}"` : "connected — grouping with devices on your network");
  ws.onclose = () => {
    if (wasDenied) return;
    setStatus("signaling disconnected — reconnecting…");
    setTimeout(connectSignaling, 3000);
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "welcome") {
      myId = msg.id;
      if (msg.room) log(`group: ${msg.room.startsWith("net:") ? "your network" : "private room " + msg.room.replace("room:", "")}`);
      if (msg.host) { log("you host this room — joiners wait for your approval"); setStatus(`hosting private room "${room()}"`); }
      else if (room()) setStatus(`accepted into private room "${room()}"`);
      for (const p of msg.peers) { names.set(p.id, p.name); if (!chans.has(p.id)) initiatePeer(p.id); }
      updatePeers();
    } else if (msg.type === "waiting") {
      setStatus("knocking — waiting for the room's host to let you in…");
    } else if (msg.type === "denied") {
      wasDenied = true; setStatus("the host declined your request to join"); log("join request declined by the host");
    } else if (msg.type === "host") {
      log("the host left — you are now the host of this room"); setStatus(`hosting private room "${room()}"`);
    } else if (msg.type === "join-request") {
      addJoinRequest(msg.id, msg.name);
    } else if (msg.type === "peer-joined") {
      names.set(msg.id, msg.name); updatePeers(); log(`${msg.name} joined`);
      // Only the NEWEST peer dials (so the two never collide with competing
      // offers), which means this side is waiting to be called. It still needs
      // a deadline: if the offer never arrives, or arrives and its ICE never
      // completes, there is no connection object here to fail — and without a
      // watchdog this side shows "(connecting…)" forever with nothing to
      // recover from. That is the failure two separate machines actually hit.
      if (forceRelay) useRelay(msg.id);
      else if (!chans.has(msg.id)) armWatchdog(msg.id);
    } else if (msg.type === "peer-left") {
      log(`${nmeOf(msg.id)} left`); cleanupPeer(msg.id); names.delete(msg.id); caps.delete(msg.id); updatePeers(); renderPlan();
    } else if (msg.type === "signal") {
      await onSignal(msg.from, msg.data);
    } else if (msg.type === "ping") {
      ws.send(JSON.stringify({ type: "pong" }));
    } else if (msg.type === "relay") {
      onWire(msg.from, bufFromB64(msg.data));
    }
  };
}
function signal(to, data) { ws.send(JSON.stringify({ type: "signal", to, data })); }

function addJoinRequest(id, name) {
  const row = document.createElement("div");
  row.className = "joinrow";
  const who = document.createElement("span");
  who.textContent = `❄ ${name} wants to join`;     // textContent: names come off the wire
  const btn = (label, allow) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.className = allow ? "small" : "small danger";
    b.onclick = () => {
      ws.send(JSON.stringify({ type: "admit", id, allow }));
      row.remove();
      log(allow ? `you let ${name} in` : `you declined ${name}`);
    };
    return b;
  };
  row.append(who, btn("Accept", true), btn("Deny", false));
  ui.requests.appendChild(row);
}

// ---- connection watchdog -----------------------------------------------------
// WebRTC can fail by never finishing. A peer whose ICE never completes sits in
// "checking" indefinitely: `dc.onclose` never fires because the channel never
// opened, and `connectionState` may never reach "failed" either. Without a
// timeout the UI shows "(connecting…)" forever, the relay fallback below is
// never reached, and nothing says why — which is exactly what happened on two
// real machines while two tabs on one machine worked fine, because same-host
// candidates always succeed.
//
// So: give each attempt a deadline, say what state it died in, retry once, then
// fall back to relaying through the server.
const CONNECT_TIMEOUT = 12000;
const MAX_DIRECT_ATTEMPTS = 2;
const watchdogs = new Map();                   // peerId -> timer
const attempts = new Map();                    // peerId -> count
const relayed = new Set();                     // peers we reach via the server

function clearWatchdog(peerId) {
  const t = watchdogs.get(peerId);
  if (t) { clearTimeout(t); watchdogs.delete(peerId); }
}
function armWatchdog(peerId) {
  clearWatchdog(peerId);
  watchdogs.set(peerId, setTimeout(() => {
    watchdogs.delete(peerId);
    if (chans.has(peerId) || !names.has(peerId)) return;
    const pc = pcs.get(peerId);
    const n = (attempts.get(peerId) || 0) + 1;
    attempts.set(peerId, n);
    log(`no direct path to ${nmeOf(peerId)} after ${CONNECT_TIMEOUT / 1000}s — ` +
        `ICE ${pc ? pc.iceConnectionState : "?"}, gathering ${pc ? pc.iceGatheringState : "?"} ` +
        `(attempt ${n} of ${MAX_DIRECT_ATTEMPTS})`);
    if (n < MAX_DIRECT_ATTEMPTS) {
      // one side redials so the two do not collide (offer glare); the other
      // just waits out another deadline
      if (+myId.slice(1) > +peerId.slice(1)) { cleanupPeer(peerId); initiatePeer(peerId); }
      else armWatchdog(peerId);
      return;
    }
    useRelay(peerId);                          // symmetric: both sides install it
  }, CONNECT_TIMEOUT));
}

// Relay through the signaling server. Both peers already hold a WebSocket to
// it, so this is a path that cannot fail for NAT reasons — but it means the
// server carries activations, which the direct path specifically avoids. That
// is a real change in what the server sees, so it is stated loudly rather than
// slipped in as a silent recovery.
function useRelay(peerId) {
  if (chans.has(peerId) || !names.has(peerId)) return;
  cleanupPeer(peerId);
  chans.set(peerId, makeRelayChannel(peerId));
  relayed.add(peerId);
  log(`⚠ falling back to the SERVER RELAY for ${nmeOf(peerId)} — no direct WebRTC path could be ` +
      `established. This works, but activations for that hop now pass through the server instead ` +
      `of peer-to-peer. Supply a TURN server (DAISY_RTC_CONFIG) for a direct path.`);
  // A relay channel is a plain object with no "open" event, so nothing here
  // fires dc.onopen — and without this the capability exchange never happens,
  // the peer never enters the plan, and the ring silently shrinks to whoever
  // had a direct channel. Losing a stage quietly is worse than failing loudly.
  sendHello(peerId);
  updatePeers(); wake();
}

function newPC(peerId) {
  const pc = new RTCPeerConnection(rtcConfig);
  pc.onicecandidate = (e) => { if (e.candidate) signal(peerId, { candidate: e.candidate }); };
  // Terse but real diagnostics: without these, a cross-machine failure is
  // indistinguishable from a hang.
  pc.oniceconnectionstatechange = () => {
    const s = pc.iceConnectionState;
    if (s === "failed" || s === "disconnected" || s === "connected" || s === "completed")
      log(`${nmeOf(peerId)}: ICE ${s}`);
  };
  pc.onicegatheringstatechange = () => {
    if (pc.iceGatheringState === "complete" && !chans.has(peerId))
      log(`${nmeOf(peerId)}: finished gathering candidates, still no channel`);
  };
  pc.onconnectionstatechange = () => {
    if (pc.connectionState === "failed") {
      log(`${nmeOf(peerId)}: connection failed`);
      clearWatchdog(peerId);
      const n = (attempts.get(peerId) || 0) + 1;
      attempts.set(peerId, n);
      cleanupPeer(peerId);
      if (n < MAX_DIRECT_ATTEMPTS) scheduleReconnect(peerId);
      else useRelay(peerId);
    }
    if (pc.connectionState === "disconnected")
      setTimeout(() => {
        if (pcs.get(peerId) === pc && pc.connectionState === "disconnected") {
          log(`${nmeOf(peerId)} connection did not recover — redialing`);
          cleanupPeer(peerId); scheduleReconnect(peerId);
        }
      }, 4000);
  };
  pcs.set(peerId, pc);
  return pc;
}

// Which path actually carried the connection — host, srflx (STUN) or relay
// (TURN). Printed on success because "it connected" and "it connected the way
// you think" are different facts.
async function logSelectedPath(peerId, pc) {
  try {
    const stats = await pc.getStats();
    let pair = null;
    stats.forEach(r => {
      if (r.type === "candidate-pair" && r.state === "succeeded" && (r.selected || r.nominated)) pair = r;
    });
    if (!pair) return;
    const loc = stats.get(pair.localCandidateId), rem = stats.get(pair.remoteCandidateId);
    log(`${nmeOf(peerId)}: direct path via ${loc ? loc.candidateType : "?"} → ${rem ? rem.candidateType : "?"}`);
  } catch (e) {}
}
function b64FromBuf(buf) {
  const u = new Uint8Array(buf); let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
}
function bufFromB64(s) { return Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer; }
function makeRelayChannel(peerId) {
  return {
    isRelay: true,
    get readyState() { return ws && ws.readyState === 1 && names.has(peerId) ? "open" : "closed"; },
    get bufferedAmount() { return ws ? ws.bufferedAmount : 0; },
    send(buf) { ws.send(JSON.stringify({ type: "relay", to: peerId, data: b64FromBuf(buf) })); },
    close() { chans.delete(peerId); },
  };
}
const reconnectTimers = new Map();
function scheduleReconnect(peerId, attempt = 0) {
  if (!names.has(peerId) || chans.has(peerId) || reconnectTimers.has(peerId)) return;
  if (attempt >= MAX_DIRECT_ATTEMPTS) return void useRelay(peerId);
  const delay = 1500 * Math.pow(2, attempt);
  reconnectTimers.set(peerId, setTimeout(() => {
    reconnectTimers.delete(peerId);
    if (!names.has(peerId) || chans.has(peerId)) return;
    if (+myId.slice(1) > +peerId.slice(1)) {
      log(`reconnecting to ${nmeOf(peerId)} (attempt ${attempt + 1})…`);
      cleanupPeer(peerId); initiatePeer(peerId);
    } else {
      armWatchdog(peerId);                       // the other side dials; still deadline it
    }
    setTimeout(() => scheduleReconnect(peerId, attempt + 1), CONNECT_TIMEOUT + 2000);
  }, delay));
}
function initiatePeer(peerId) {
  if (forceRelay) return useRelay(peerId);       // ?relay=1 — skip WebRTC entirely
  const pc = newPC(peerId);
  setupChannel(peerId, pc.createDataChannel("daisy"));
  armWatchdog(peerId);                           // no silent forever-connecting
  pc.createOffer().then(o => pc.setLocalDescription(o)).then(() => signal(peerId, { sdp: pc.localDescription }));
}
const pendingCand = new Map();
async function onSignal(from, data) {
  let pc = pcs.get(from);
  if (data.sdp) {
    if (!pc) {
      pc = newPC(from);
      pc.ondatachannel = (e) => setupChannel(from, e.channel);
      armWatchdog(from);                         // the answering side needs a deadline too
    }
    await pc.setRemoteDescription(data.sdp);
    for (const c of pendingCand.get(from) || []) try { await pc.addIceCandidate(c); } catch (e) {}
    pendingCand.delete(from);
    if (data.sdp.type === "offer") {
      const ans = await pc.createAnswer(); await pc.setLocalDescription(ans);
      signal(from, { sdp: pc.localDescription });
    }
  } else if (data.candidate) {
    if (pc && pc.remoteDescription) { try { await pc.addIceCandidate(data.candidate); } catch (e) {} }
    else { if (!pendingCand.has(from)) pendingCand.set(from, []); pendingCand.get(from).push(data.candidate); }
  }
}
function setupChannel(peerId, dc) {
  dc.binaryType = "arraybuffer";
  dc.onopen = () => {
    clearWatchdog(peerId);
    attempts.delete(peerId);
    relayed.delete(peerId);
    chans.set(peerId, dc); updatePeers();
    log(`connected to ${nmeOf(peerId)}`);
    const pc = pcs.get(peerId);
    if (pc) logSelectedPath(peerId, pc);
    sendHello(peerId);
  };
  dc.onclose = () => { chans.delete(peerId); updatePeers(); wake(); scheduleReconnect(peerId); };
  dc.onmessage = (e) => onWire(peerId, e.data);
}
function cleanupPeer(id) {
  clearWatchdog(id);
  const pc = pcs.get(id); if (pc) pc.close();
  pcs.delete(id); pendingCand.delete(id);
  const dc = chans.get(id);
  if (dc && !dc.isRelay) chans.delete(id);       // keep a working relay channel
  for (const k of fragIn.keys()) if (k.startsWith(id + ":")) fragIn.delete(k);
  wake();
}

// ---- fragmentation ----------------------------------------------------------
const FRAG_SENTINEL = Wire.FRAG, FRAG_CHUNK = 48 * 1024, DC_MAXBUF = 4 * 1024 * 1024;
let fragSeq = 1;
const fragIn = new Map();
function dcDrain(dc) {
  return new Promise((res) => {
    if (dc.bufferedAmount <= DC_MAXBUF || dc.readyState !== "open") return res();
    const t = setInterval(() => {
      if (dc.bufferedAmount <= DC_MAXBUF || dc.readyState !== "open") { clearInterval(t); res(); }
    }, 50);
  });
}
async function dcSend(dc, buf) {
  if (buf.byteLength <= FRAG_CHUNK) {
    await dcDrain(dc);
    if (dc.readyState === "open") dc.send(buf);
    return;
  }
  const id = fragSeq++, src = new Uint8Array(buf);
  const total = Math.ceil(src.length / FRAG_CHUNK);
  for (let s = 0; s < total; s++) {
    await dcDrain(dc);
    if (dc.readyState !== "open") return;
    const part = src.subarray(s * FRAG_CHUNK, Math.min((s + 1) * FRAG_CHUNK, src.length));
    const msg = new ArrayBuffer(16 + part.length);
    new Int32Array(msg, 0, 4).set([FRAG_SENTINEL, id, s, total]);
    new Uint8Array(msg, 16).set(part);
    dc.send(msg);
  }
}
function onFragment(peerId, buf) {
  const [, id, seq, total] = new Int32Array(buf, 0, 4);
  const key = peerId + ":" + id;
  let st = fragIn.get(key);
  if (!st) { st = { id, parts: [], got: 0, total }; fragIn.set(key, st); }
  st.parts[seq] = new Uint8Array(buf, 16).slice(0);
  st.got++;
  if (st.got < st.total) return null;
  fragIn.delete(key);
  let len = 0; for (const p of st.parts) len += p.length;
  const out = new Uint8Array(len);
  let off = 0; for (const p of st.parts) { out.set(p, off); off += p.length; }
  return out.buffer;
}
function sendTo(peerId, buf) {
  const dc = chans.get(peerId);
  if (!dc || dc.readyState !== "open") return Promise.resolve(false);
  return dcSend(dc, buf).then(() => true);
}
function broadcast(buf) {
  return Promise.all([...chans.values()].filter(dc => dc.readyState === "open").map(dc => dcSend(dc, buf)));
}

// ---- hello / capability -----------------------------------------------------
function sendHello(peerId) {
  return sendTo(peerId, Wire.packHello(myCapacity, probeHash, compute ? compute.backend : "?"));
}
function onHello(peerId, buf) {
  let h;
  try { h = Wire.unpackHello(buf); } catch (e) { log(`bad hello from ${nmeOf(peerId)}: ${e.message}`); return; }
  caps.set(peerId, { capacity: h.capacity, backend: h.backend, probe: h.probeHash });
  // The kernel probe matters MORE in a pipeline than in the trainer. There,
  // every peer computes the same thing, so bad arithmetic shows up as a
  // diverging replica. Here each stage computes something different and nobody
  // repeats it — a broken middle stage would corrupt every token invisibly.
  // The probe is the one value that stays comparable when the work is not.
  if (probeHash && h.probeHash && h.probeHash !== probeHash)
    log(`⚠ ${nmeOf(peerId)} disagrees with this device's kernel probe (${h.probeHash} vs ${probeHash}). ` +
        `Their arithmetic differs from ours — do not put them in the ring.`);
  renderPlan();
}

// ---- capacity ---------------------------------------------------------------
let myCapacity = 1;
async function measureCapacity() {
  const m = 32, k = 64, n = 64;
  const X = new Float32Array(m * k), W = new Float32Array(k * n);
  for (let i = 0; i < X.length; i++) X[i] = Math.sin(i) * 0.5;
  for (let i = 0; i < W.length; i++) W[i] = Math.cos(i) * 0.5;
  await Verified.vgemmBlock(X, W, { m, k, n, batch: 1 }, L, compute.bgemm, null);
  const t0 = performance.now();
  let it = 0;
  while (performance.now() - t0 < 300) { await Verified.vgemmBlock(X, W, { m, k, n, batch: 1 }, L, compute.bgemm, null); it++; }
  myCapacity = it / ((performance.now() - t0) / 1000);
  return myCapacity;
}

// ---- loading a model from the Hub -------------------------------------------
// Only the header and config are read here: a few tens of KB, regardless of how
// large the model is. Weights are fetched per stage, later, by each device.
async function loadRepo() {
  const repo = ui.repo.value.trim();
  const revision = (ui.revision.value.trim() || "main");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) { log(`"${repo}" is not a valid repo id (expected owner/name)`); return; }
  ui.loadRepo.disabled = true;
  ui.modelState.textContent = "reading config…";
  try {
    const cfg = await withAuth((t) => HF.getJSON(repo, "config.json", revision, t), `${repo}/config.json`);
    const s = Arch.fromConfig(cfg);
    ui.modelState.textContent = "reading tokenizer…";
    const tokJson = await withAuth((t) => HF.getJSON(repo, "tokenizer.json", revision, t), `${repo}/tokenizer.json`);
    tok = Tokenizer.build(tokJson);
    ui.tokenizer.textContent = tok.name;
    ui.modelState.textContent = "reading weight index…";
    const { tensors } = await withAuth((t) => HF.readIndex(repo, revision, t, (m) => { ui.modelState.textContent = m; }),
                                       `${repo} weights`);
    const fingerprint = Shard.modelFingerprint(repo, revision, tensors);
    repoInfo = { repo, revision, spec: s, tensors, fingerprint };
    spec = s; modelHash = fingerprint;
    const totalMB = [...tensors.values()].reduce((a, t) => a + t.elems * 4, 0) / 1048576;
    ui.model.textContent =
      `${repo} · ${s.family}-style · ${s.layers} layers · hidden ${s.hidden} · ` +
      `heads ${s.heads}${s.kvHeads !== s.heads ? "/" + s.kvHeads + " kv" : ""} · vocab ${s.vocab}`;
    ui.modelState.textContent = `${totalMB.toFixed(0)} MB in f32 · fingerprint ${fingerprint.toString(16)}`;
    log(`model ready: ${repo}@${revision} — ${s.layers} layers, ${totalMB.toFixed(0)} MB total in f32. ` +
        `Nothing has been downloaded yet beyond headers; each device will fetch only its own layers.`);
    if (s.maxPos < ctxLen()) log(`⚠ this model's max position is ${s.maxPos} — lower the context length`);
    ui.genBtn.disabled = false;
    ui.verifyRow.style.display = "";
    renderPlan();
  } catch (e) {
    ui.modelState.textContent = "failed";
    log(`could not load ${repo}: ${redact(e.message)}`);
  }
  ui.loadRepo.disabled = false;
}

function buildPlan() {
  if (!repoInfo) return null;
  const list = [{ id: myId, capacity: myCapacity, backend: compute.backend }];
  for (const id of [...chans.keys()].sort((a, b) => +a.slice(1) - +b.slice(1))) {
    const c = caps.get(id);
    if (c) list.push({ id, capacity: c.capacity, backend: c.backend });
  }
  return Shard.planStages(repoInfo.spec, list);
}
function renderPlan() {
  if (!plan) {
    const known = 1 + [...chans.keys()].filter(id => caps.has(id)).length;
    ui.plan.textContent = repoInfo ? `${known} device(s) ready — press Generate to shard and run`
                                   : "load a model to see how it would be split";
    return;
  }
  ui.plan.textContent = plan.map(s => {
    const who = s.id === myId ? "me" : nmeOf(s.id);
    const part = s.hi > s.lo ? `layers ${s.lo}–${s.hi - 1}` : "—";
    let mb = "";
    if (repoInfo) mb = " · " + (Shard.stageBytes(repoInfo.spec, repoInfo.tensors, s, Arch) / 1048576).toFixed(0) + " MB";
    return `${s.index}. ${who}${s.head ? " (head: embed + lm_head)" : ""} · ${part}${mb}` +
           (readyStages.has(s.index) ? " ✓" : "");
  }).join("\n");
}

// Fetch this device's own layers. This is the only place weight bytes move,
// and they move from the Hub to here — never from a peer.
async function loadMyStage(assign, st) {
  const s = assign.spec;
  ui.myShard.textContent = "reading index…";
  const { tensors } = await withAuth((t) => HF.readIndex(assign.repo, assign.revision, t), `${assign.repo} weights`);
  const want = Arch.tensorsFor(s, tensors, st).map(n => tensors.get(n));
  const mb = want.reduce((a, t) => a + t.elems * 4, 0) / 1048576;
  log(`fetching my slice from the Hub: ${want.length} tensors, ${mb.toFixed(0)} MB ` +
      `(the other ${s.layers - (st.hi - st.lo)} layers are never downloaded here)`);
  const got = await withAuth((t) => HF.fetchTensors(assign.repo, assign.revision, t, want,
                                                    (p) => { ui.myShard.textContent = `downloading ${p}`; }),
                             `${assign.repo} weights`);
  const w = Shard.stageWeights(s, st, got, Arch);
  stageObj = Infer.makeStage(s, st, w, kernelCtx(), assign.ctx || ctxLen());
  spec = s;
  ui.myShard.textContent = `${st.head ? "head + " : ""}layers ${st.lo}–${st.hi - 1} · ${mb.toFixed(0)} MB resident`;
  return stageObj;
}

async function onAssign(peerId, buf) {
  if (running) { log(`ignored an assignment from ${nmeOf(peerId)} mid-run`); return; }
  let a;
  try { a = Wire.unpackAssign(buf); } catch (e) { log(`REFUSED assignment from ${nmeOf(peerId)}: ${e.message}`); return; }
  plan = a.plan; spec = a.spec; modelHash = a.fingerprint;
  myStage = plan[a.mine];
  log(`assigned by ${nmeOf(peerId)}: ${a.repo}@${a.revision}, layers ${myStage.lo}–${myStage.hi - 1} ` +
      `of ${a.spec.layers} (stage ${a.mine} of ${plan.length})`);
  renderPlan();
  try {
    // tokenizer only matters on the head, but loading it is cheap and lets a
    // middle stage show the token stream
    if (!tok) {
      try { tok = Tokenizer.build(await withAuth((t) => HF.getJSON(a.repo, "tokenizer.json", a.revision, t), "tokenizer")); }
      catch (e) { log(`(no tokenizer here: ${redact(e.message)} — this stage will show ids, not text)`); }
    }
    await loadMyStage(a, myStage);
    await sendTo(peerId, Wire.packReady(a.mine, true, "loaded"));
    log(`ready — holding layers ${myStage.lo}–${myStage.hi - 1}. The rest of the model is not on this device.`);
    setStatus("in the ring — waiting for work");
  } catch (e) {
    await sendTo(peerId, Wire.packReady(a.mine, false, redact(e.message)));
    log(`could not take my slice: ${redact(e.message)}`);
  }
}
function onReady(peerId, buf) {
  const r = Wire.unpackReady(buf);
  if (r.ok) { readyStages.add(r.stageIndex); log(`${nmeOf(peerId)} is ready (stage ${r.stageIndex})`); }
  else log(`${nmeOf(peerId)} could not load stage ${r.stageIndex}: ${r.note}`);
  renderPlan(); wake();
}

// ---- the ring ---------------------------------------------------------------
// One token = one lap. The head embeds the window and hands the hidden state
// to stage 1; each stage runs its own layers and hands the result on; the last
// stage returns it to the head, which turns it into a token.
//
// The payload is T x hidden floats — tens of KB. The weights it stands in for
// are hundreds of megabytes. That asymmetry is the whole reason this works,
// and it makes a run latency-bound rather than bandwidth-bound.
let runSeq = 0;
const waiters = new Set();
function wake() { for (const w of waiters) w(); }
function waitFor(pred, timeoutMs) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    let timer = null;
    const check = () => {
      const ok = pred();
      if (ok || Date.now() - t0 > timeoutMs) { waiters.delete(check); clearInterval(timer); resolve(!!ok); }
    };
    waiters.add(check);
    timer = setInterval(check, 200);
    check();
  });
}
const actIn = new Map();

function packAct(seq, tokenIdx, nextIndex, hidden) {
  return Wire.packAct(seq, tokenIdx, nextIndex, hidden, Shard.hashF32, modelHash);
}
async function onAct(peerId, buf) {
  let a;
  try { a = Wire.unpackAct(buf, Shard.hashF32); }
  catch (e) { log(`activation from ${nmeOf(peerId)} rejected: ${e.message}`); return; }
  const { seq, tokenIdx, nextIndex, modelHash: mh, hidden } = a;
  if (mh !== modelHash) {
    log(`REFUSED activation from ${nmeOf(peerId)}: it belongs to model ${mh.toString(16)}, this device ` +
        `holds a slice of ${modelHash.toString(16)}. Mixing two models would produce confident nonsense.`);
    return;
  }
  const me = plan && plan.find(s => s.id === myId);
  const kind = Wire.classifyAct(nextIndex, me ? me.index : -99);
  if (kind === "return") { actIn.set(`${seq}:${tokenIdx}`, hidden); wake(); return; }
  if (kind === "other" || !me || !stageObj) return;
  const t0 = performance.now();
  let x;
  try { x = await Infer.runLayers(stageObj, hidden); }
  catch (e) { log(`stage failed: ${redact(e.message)}`); return; }
  if (auditFailure) { log("KERNEL AUDIT FAILED mid-hop — refusing to pass on a value I cannot vouch for"); return; }
  const hop = Wire.routeAfter(plan, me.index);
  ui.ringState.textContent = `layers ${me.lo}–${me.hi - 1} in ${(performance.now() - t0).toFixed(0)} ms → ` +
                             `${hop.to === myId ? "me" : nmeOf(hop.to)}${hop.isReturn ? " (return)" : ""}`;
  await sendTo(hop.to, packAct(seq, tokenIdx, hop.address, x));
}

async function generate() {
  if (running) return;
  if (!repoInfo) { log("load a model first — the device that loads it drives the ring"); return; }
  if (!tok) { log("no tokenizer loaded"); return; }
  running = true; abortRun = false; auditFailure = null;
  ui.genBtn.disabled = true; ui.stopBtn.disabled = false;
  readyStages.clear();
  try {
    const T = Math.min(ctxLen(), repoInfo.spec.maxPos);
    plan = buildPlan(); modelHash = repoInfo.fingerprint; spec = repoInfo.spec;
    renderPlan();
    if (plan.length > 1) {
      log(`plan: ${spec.layers} layers over ${plan.length} device(s) — ` +
          plan.map(s => `${s.id === myId ? "me" : nmeOf(s.id)}:${s.hi - s.lo}`).join(", "));
      for (let i = 1; i < plan.length; i++)
        await sendTo(plan[i].id, Wire.packAssign({
          repo: repoInfo.repo, revision: repoInfo.revision, spec, plan,
          mine: i, fingerprint: repoInfo.fingerprint, ctx: T,
        }));
      log("waiting for every stage to fetch its layers…");
      const ok = await waitFor(() => readyStages.size >= plan.length - 1, 600000);
      if (!ok) { log("not every stage reported ready — generating anyway will stall, so stopping here."); throw new Error("stages not ready"); }
    } else {
      log(`solo run — this device will hold all ${spec.layers} layers`);
    }
    myStage = plan[0];
    ui.myShard.textContent = "loading my slice…";
    await loadMyStage({ repo: repoInfo.repo, revision: repoInfo.revision, spec, ctx: T }, myStage);

    const seq = ++runSeq;
    const nTok = +ui.ntok.value, temp = +ui.temp.value / 100;
    const opts = { temperature: temp, topK: 40, rng: Infer.mulberry32(+ui.seed.value | 0) };
    let ids = [...Tokenizer.encode(tok, ui.prompt.value || "The")];
    if (!ids.length) ids = [0];
    const promptLen = ids.length;
    ui.out.textContent = ui.prompt.value;
    const tStart = performance.now();
    let hops = 0;

    for (let n = 0; n < nTok && !abortRun; n++) {
      const win = new Int32Array(T);
      const tail = ids.slice(-T);
      for (let i = 0; i < tail.length; i++) win[T - tail.length + i] = tail[i];
      let x = Infer.embed(stageObj, win);
      x = await Infer.runLayers(stageObj, x);
      if (auditFailure) { log(`KERNEL AUDIT FAILED: ${auditFailure} — stopping`); break; }
      if (plan.length > 1) {
        await sendTo(plan[1].id, packAct(seq, n, plan[1].index, x));
        hops += plan.length;
        const key = `${seq}:${n}`;
        const ok = await waitFor(() => actIn.has(key), 60000);
        if (!ok) { log(`the ring stalled at token ${n + 1}: no activation came back. Press Generate again to re-plan.`); break; }
        x = actIn.get(key); actIn.delete(key);
      }
      const id = Infer.pickToken(await Infer.readout(stageObj, x), opts);
      ids.push(id);
      ui.out.textContent = Tokenizer.decode(tok, ids);
      ui.tokcount.textContent = String(n + 1);
      ui.hops.textContent = String(hops);
      ui.tps.textContent = ((n + 1) / ((performance.now() - tStart) / 1000)).toFixed(2);
      broadcast(Wire.packToken(seq, id, ids.length));
      await new Promise(r => setTimeout(r, 0));
    }
    // The differential check. In a pipeline nothing recomputes anything, so
    // this is the only instrument that can show a distributed answer is RIGHT
    // rather than merely self-consistent. It costs a full local run, which is
    // only possible when this device can hold the whole model — so it is
    // offered, not assumed.
    if (ui.verify.checked && !abortRun && plan.length > 1) {
      log("verifying: fetching the whole model here and re-running the same prompt…");
      const all = { id: myId, index: 0, lo: 0, hi: spec.layers, head: true };
      const solo = await loadMyStage({ repo: repoInfo.repo, revision: repoInfo.revision, spec, ctx: T }, all);
      const ref = await Infer.generateLocal([solo], Tokenizer.encode(tok, ui.prompt.value || "The"), nTok,
                                            { temperature: temp, topK: 40, rng: Infer.mulberry32(+ui.seed.value | 0) });
      const same = ref.length === ids.length && ref.every((v, i) => v === ids[i]);
      log(same ? "VERIFIED: the distributed run and the single-device run produced identical token ids."
               : `MISMATCH: the ring and this device disagree. Distributed "${Tokenizer.decode(tok, ids).slice(0, 60)}" ` +
                 `vs local "${Tokenizer.decode(tok, ref).slice(0, 60)}". Check the log for probe warnings.`);
    }
    if (!abortRun) log(`done — ${ids.length - promptLen} token(s) in ${((performance.now() - tStart) / 1000).toFixed(1)} s ` +
                       `over ${plan.length} stage(s)`);
    broadcast(Wire.packDone(seq));
  } catch (e) {
    log(`run failed: ${redact(e.message)}`);
    console.error(e);
  }
  running = false;
  ui.genBtn.disabled = false; ui.stopBtn.disabled = true;
  ui.ringState.textContent = "idle";
}

function onToken(peerId, buf) {
  const { id } = Wire.unpackToken(buf);
  if (tok) ui.out.textContent += Tokenizer.decode(tok, [id]);
  else ui.out.textContent += ` ${id}`;
  ui.tokcount.textContent = String(+(ui.tokcount.textContent || 0) + 1);
}

function onWire(peerId, buf) {
  const tag = Wire.tagOf(buf);
  if (tag === Wire.FRAG) { const whole = onFragment(peerId, buf); if (whole) onWire(peerId, whole); return; }
  if (tag === Wire.HELLO) return onHello(peerId, buf);
  if (tag === Wire.ASSIGN) return void onAssign(peerId, buf);
  if (tag === Wire.READY) return onReady(peerId, buf);
  if (tag === Wire.ACT) return void onAct(peerId, buf);
  if (tag === Wire.TOKEN) return onToken(peerId, buf);
  if (tag === Wire.DONE) { ui.ringState.textContent = "idle"; return; }
}

// ---- kernels ----------------------------------------------------------------
const gemmAudit = {
  cells: 12,
  due: () => true,
  fail: (msg) => { if (!auditFailure) { auditFailure = msg; log(`KERNEL AUDIT FAILED: ${msg}`); } },
};
function kernelCtx() {
  return { L, bgemm: compute.bgemm || null, att: compute.att || null,
           mlp: compute.mlp || null, audit: gemmAudit };
}

// ---- boot -------------------------------------------------------------------
(async function () {
  try {
    const mode = await (await fetch("mode")).json();
    if (mode.rtc) rtcConfig = mode.rtc;
    oauthAvailable = !!mode.oauth;
    // Room-first deployments (the hosted Space sets DAISY_FORCE_ROOMS): there
    // is no LAN auto-grouping between strangers. Two people behind the same
    // CGNAT or on the same campus network share a public IP, and without this
    // they would be dropped into one ring together — able to see each other's
    // addresses and each other's activations. So a visitor with no room code
    // chooses: create their own room, or join one by code.
    if (mode.forceRooms && !room()) {
      for (const c of document.querySelectorAll(".card")) if (c !== ui.lobby) c.style.display = "none";
      ui.lobby.style.display = "";
      ui.createRoom.onclick = () => {
        const code = (ADJ[Math.floor(Math.random() * ADJ.length)] + "-" +
                      NOUN[Math.floor(Math.random() * NOUN.length)] + "-" +
                      (100 + Math.floor(Math.random() * 900))).toLowerCase();
        location.href = "?room=" + code;
      };
      const join = () => {
        const code = ui.joinCode.value.trim().toLowerCase();
        if (code) location.href = "?room=" + encodeURIComponent(code);
      };
      ui.joinRoom.onclick = join;
      ui.joinCode.onkeydown = (e) => { if (e.key === "Enter") join(); };
      return;                                    // wait for the choice
    }
  } catch (e) {}                                 // no /mode: local defaults
  adoptTokenFromFragment();                      // returning from an OAuth sign-in

  // Same rule as the trainer: the verified units are mandatory. No float path.
  try {
    L = await Compute.loadLUTs();
    if (!(L.mul instanceof Int16Array) || L.mul.length !== 65536) throw new Error("mul8 LUT malformed");
    if (L.mul[((7 & 0xFF) * 256) + (-3 & 0xFF)] !== -21) throw new Error("mul8 LUT self-test failed (7 × -3 ≠ -21)");
    compute = await Compute.initCompute(L);
  } catch (e) {
    setStatus("NEURAL UNITS UNAVAILABLE — inference disabled");
    ui.backend.textContent = "unavailable";
    log(`FATAL: verified neural units failed to load (${e.message}). This build only computes through the units.`);
    return;
  }
  ui.backend.textContent = `${compute.backend.toUpperCase()} — ${compute.label}`;
  probeHash = await Compute.kernelProbe(compute, L);
  log(`kernel probe: ${probeHash} — every honest device gets this same number, on any backend`);

  await measureCapacity();
  log(`measured capacity: ${myCapacity.toFixed(0)} verified GEMMs/sec — this decides how many layers this device takes`);

  ui.me.textContent = deviceName;
  if (!hfToken) setToken(null);                  // keep a token adopted from the fragment
  if (oauthAvailable) log("this deployment uses Hugging Face sign-in for gated models — no token is ever typed into this page");
  for (const [el, out] of [[ui.temp, ui.vtemp], [ui.ntok, ui.vntok], [ui.ctxLen, ui.vctxLen]])
    el.oninput = () => out.textContent = el.value;
  ui.temp.oninput(); ui.ntok.oninput(); ui.ctxLen.oninput();

  if (room()) {
    ui.roomInfo.style.display = "";
    ui.roomCode.textContent = room();
    ui.copyLink.onclick = async () => {
      try { await navigator.clipboard.writeText(location.href); ui.copyLink.textContent = "Copied!"; }
      catch { ui.copyLink.textContent = location.href; }
      setTimeout(() => ui.copyLink.textContent = "Copy invite link", 2000);
    };
  }
  ui.loadRepo.onclick = loadRepo;
  ui.repo.onkeydown = (e) => { if (e.key === "Enter") loadRepo(); };
  ui.clearToken.onclick = clearToken;
  ui.genBtn.onclick = generate;
  ui.stopBtn.onclick = () => { abortRun = true; log("stopping after the current token"); };

  updatePeers(); renderPlan();
  connectSignaling();
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && (!ws || ws.readyState > 1)) { log("page woke — reconnecting"); connectSignaling(); }
  });
  if (navigator.connection && navigator.connection.addEventListener)
    navigator.connection.addEventListener("change", () => {
      if (!ws || ws.readyState > 1) { log("network changed — reconnecting"); connectSignaling(); }
    });
  setStatus("ready — load a model to drive, or wait to be given a slice");
  log(`ready — this device is "${deviceName}", computing through verified units on ${compute.backend.toUpperCase()}`);
})();
