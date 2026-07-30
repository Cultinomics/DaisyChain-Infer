// DaisyChain-Infer signaling + static host.
//
// Deliberately tiny. What it does NOT do is the point: it introduces peers and
// relays WebRTC handshakes, and never sees a weight, an activation, a prompt,
// or a generated token. Those move directly between the devices in the ring,
// and weights come from the Hugging Face CDN straight to each browser.
//
// The one exception is the OAuth code exchange below, which by construction
// has to handle an access token for a few milliseconds. That is called out
// rather than glossed: see the comment on /auth/callback.
//
// Only dependency: `ws`.  Run:  npm install && npm start
const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { WebSocketServer } = require("ws");

// ---- rooms -------------------------------------------------------------------
// Snapdrop-style grouping: same public IP -> same room, so devices on your own
// network find each other with no code. ?room=CODE crosses networks, gated by
// host approval.
function clientIP(req) {
  const xff = req.headers["x-forwarded-for"];
  if (xff) return xff.split(",")[0].trim();
  return (req.socket.remoteAddress || "unknown").replace(/^::ffff:/, "");
}
function roomFor(req) {
  const u = new URL(req.url, "http://x");
  const code = u.searchParams.get("room");
  if (code) return "room:" + code;
  const h = crypto.createHash("sha256").update(clientIP(req)).digest("hex").slice(0, 10);
  return "net:" + h;
}

const PORT = process.env.PORT || 8788;
const PUB = path.join(__dirname, "public");
const TYPES = { ".html": "text/html", ".js": "text/javascript",
                ".css": "text/css", ".json": "application/json" };

// ---- OAuth (hosted deployments) ----------------------------------------------
// On a Hugging Face Space with `hf_oauth: true`, these are injected by the
// platform. When they are absent — a local run — OAuth is simply off and the
// page falls back to asking for a token directly, which is safe there because
// the page is served from the user's own machine.
const OA = {
  clientId: process.env.OAUTH_CLIENT_ID,
  clientSecret: process.env.OAUTH_CLIENT_SECRET,
  scopes: process.env.OAUTH_SCOPES || "read-repos",
  provider: (process.env.OPENID_PROVIDER_URL || "https://huggingface.co").replace(/\/$/, ""),
  spaceHost: process.env.SPACE_HOST || "",
};
const oauthOn = !!(OA.clientId && OA.clientSecret);

// One-time `state` values, so a callback cannot be replayed or forged. Short
// TTL, and consumed on use.
const states = new Map();                 // state -> expiry ms
function newState() {
  const s = crypto.randomBytes(24).toString("base64url");
  states.set(s, Date.now() + 10 * 60 * 1000);
  return s;
}
function takeState(s) {
  const exp = states.get(s);
  if (exp === undefined) return false;
  states.delete(s);                       // single use
  return exp > Date.now();
}
setInterval(() => {
  const now = Date.now();
  for (const [s, exp] of states) if (exp <= now) states.delete(s);
}, 60000);

function redirectUri(req) {
  const host = OA.spaceHost || req.headers.host;
  const proto = OA.spaceHost || /^(localhost|127\.|\[::1\])/.test(host || "") === false ? "https" : "http";
  return `${proto}://${host}/auth/callback`;
}

function oauthLogin(req, res) {
  const state = newState();
  const u = new URL(OA.provider + "/oauth/authorize");
  u.searchParams.set("client_id", OA.clientId);
  u.searchParams.set("redirect_uri", redirectUri(req));
  u.searchParams.set("response_type", "code");
  u.searchParams.set("scope", OA.scopes);
  u.searchParams.set("state", state);
  res.writeHead(302, { Location: u.toString() });
  res.end();
}

// The token exchange needs the client secret, so it MUST happen here rather
// than in the browser — which means this server briefly holds an access token.
// It is never written to disk, never logged, and never kept: it is handed to
// the browser in the URL *fragment*, which browsers do not send to servers and
// which the page strips from history immediately on arrival. That is the
// smallest exposure the OAuth flow allows; it is not zero, and the Space's
// README says so.
async function oauthCallback(req, res, url) {
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const err = url.searchParams.get("error");
  const bail = (msg) => {
    res.writeHead(302, { Location: "/#oauth_error=" + encodeURIComponent(msg) });
    res.end();
  };
  if (err) return bail(err);
  if (!code || !state) return bail("missing code or state");
  if (!takeState(state)) return bail("state was not recognised — start the sign-in again");
  try {
    const body = new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri(req),
      client_id: OA.clientId,
      client_secret: OA.clientSecret,
    });
    const r = await fetch(OA.provider + "/oauth/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
      body,
    });
    if (!r.ok) return bail(`token exchange failed (HTTP ${r.status})`);
    const j = await r.json();
    if (!j.access_token) return bail("no access token in the response");
    res.writeHead(302, { Location: "/#hf=" + encodeURIComponent(j.access_token) });
    res.end();
  } catch (e) {
    bail("token exchange error");                 // deliberately not e.message
  }
}

// keepalive: proxies close idle WebSockets, and a ring can sit idle between
// prompts far longer than a training run ever does
setInterval(() => {
  for (const room of rooms.values())
    for (const [, v] of room.peers) {
      if (v.ws.isAlive === false) { v.ws.terminate(); continue; }
      v.ws.isAlive = false;
      send(v.ws, { type: "ping" });
    }
}, 30000);

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  let p = decodeURIComponent(url.pathname);

  if (p === "/mode") {
    let rtc = null;
    try { if (process.env.DAISY_RTC_CONFIG) rtc = JSON.parse(process.env.DAISY_RTC_CONFIG); } catch (e) {}
    res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    return res.end(JSON.stringify({
      // DAISY_FORCE_ROOMS=1 (set on the hosted Space): no LAN auto-grouping, so
      // strangers who happen to share a public IP — same CGNAT, same campus,
      // same office — are never put in one ring. Each visitor makes their own
      // room and invites devices by link.
      forceRooms: !!process.env.DAISY_FORCE_ROOMS,
      // When OAuth is available the page does not offer a paste-a-token box at
      // all: on a deployment the user does not control, signing in is the only
      // credential path offered.
      oauth: oauthOn,
      rtc,
    }));
  }
  if (p === "/auth/login" || p === "/oauth/login") {
    if (!oauthOn) { res.writeHead(404); return res.end("oauth is not configured"); }
    return oauthLogin(req, res);
  }
  // HF registers /auth/callback for Spaces; the alias costs nothing and saves a
  // broken deploy if the platform ever hands back the other spelling.
  if (p === "/auth/callback" || p === "/oauth/callback") {
    if (!oauthOn) { res.writeHead(404); return res.end("oauth is not configured"); }
    return void oauthCallback(req, res, url);
  }

  if (p === "/") p = "/index.html";
  const file = path.join(PUB, path.normalize(p));
  if (!file.startsWith(PUB)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
});

const wss = new WebSocketServer({ server });
const rooms = new Map();
let nextId = 1;
function send(ws, obj) { if (ws.readyState === 1) ws.send(JSON.stringify(obj)); }

wss.on("connection", (ws, req) => {
  const roomId = roomFor(req);
  const name = (new URL(req.url, "http://x").searchParams.get("name") || "").slice(0, 40) || ("p" + nextId);
  const id = "p" + (nextId++);
  ws.peerId = id; ws.roomId = roomId;
  if (!rooms.has(roomId)) rooms.set(roomId, { peers: new Map(), host: null, pending: new Map() });
  const room = rooms.get(roomId);
  const isPrivate = roomId.startsWith("room:");

  function admit(pid, peer) {
    const roster = [...room.peers.entries()].map(([qid, v]) => ({ id: qid, name: v.name }));
    send(peer.ws, { type: "welcome", id: pid, room: roomId, peers: roster, host: room.host === pid });
    for (const [, v] of room.peers) send(v.ws, { type: "peer-joined", id: pid, name: peer.name });
    room.peers.set(pid, peer);
  }

  if (isPrivate && room.peers.size === 0) room.host = id;
  if (isPrivate && room.host !== id) {
    room.pending.set(id, { ws, name });
    send(ws, { type: "waiting" });
    const h = room.peers.get(room.host);
    if (h) send(h.ws, { type: "join-request", id, name });
  } else {
    admit(id, { ws, name });
  }

  ws.on("message", (buf) => {
    let msg; try { msg = JSON.parse(buf); } catch { return; }
    if (msg.type === "signal" && msg.to && room.peers.has(id)) {
      const target = room.peers.get(msg.to);
      if (target) send(target.ws, { type: "signal", from: id, data: msg.data });
    } else if (msg.type === "pong") {
      ws.isAlive = true;
    } else if (msg.type === "relay" && msg.to && room.peers.has(id)) {
      // last-resort path when two devices cannot form a direct WebRTC route.
      // Note this DOES put activations through the server — the UI says so.
      const target = room.peers.get(msg.to);
      if (target) send(target.ws, { type: "relay", from: id, data: msg.data });
    } else if (msg.type === "admit" && id === room.host) {
      const p = room.pending.get(msg.id);
      if (!p) return;
      room.pending.delete(msg.id);
      if (msg.allow) admit(msg.id, p);
      else { send(p.ws, { type: "denied" }); p.ws.close(); }
    }
  });

  ws.on("close", () => {
    room.pending.delete(id);
    if (room.peers.delete(id))
      for (const [, v] of room.peers) send(v.ws, { type: "peer-left", id });
    if (room.host === id) {
      room.host = room.peers.keys().next().value ?? null;
      const h = room.peers.get(room.host);
      if (h) {
        send(h.ws, { type: "host" });
        for (const [pid, p] of room.pending) send(h.ws, { type: "join-request", id: pid, name: p.name });
      }
    }
    if (room.peers.size === 0 && room.pending.size === 0) rooms.delete(roomId);
  });
});

server.listen(PORT, () =>
  console.log(`DaisyChain-Infer on http://localhost:${PORT}` +
              (oauthOn ? " (HF OAuth enabled)" : "") +
              (process.env.DAISY_FORCE_ROOMS ? " (rooms forced)" : "")));
