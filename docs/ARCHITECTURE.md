# Architecture

How a page load becomes one stage of a model, and how a token gets produced by
several machines that each hold only part of it.

## Added 2026-10-22

The rest of this file is the previous architecture note. This list is what changed.

- A disagreeing kernel probe is removed before apportionment (`ringMembers`).
- `estimatePlanMs` is the lap budget: each stage's share at that stage's rate. Above 120000 ms the run is refused.
- Assignments carry `pass`: `chunk` or `window`. Stages do not infer the path from the row count.
- The next weight range is fetched while the previous body is widened to f32.
- Full-window laps can score a prompt-lookup draft of at most 4 tokens in one window.
- Prompts longer than 32 tokens on more than one stage use even chunks of about 16, two in flight, with per-stage KV. Not the full-window GEMM.
- A second prompt is queued. A dropped planned peer or a stalled lap replays the prompt once. A non-finite hidden state is not forwarded.

## Added 2026-10-23

- `makeParamShard` packs a stage's layers into one flat Float32 buffer when the stage owns two or more. `runLayers` and `runLayersChunk` gather the current layer, prefetch the next, and release the one just used. The limit is 2 live layers.
- Head tensors stay outside that buffer. A one-layer stage is not sharded. No optimizer state is stored. No parameter is sent to a peer.
- A configured sliding window is a softmax mask. Key columns stay in the GEMM. `swaFrom` is `max_window_layers`.
- `seqDraw(seed, pos)` samples one position. `pickStructured` applies a JSON mask to the top 40 logits on the head. `preparedIds` is the prompt tokenized during the previous turn.
- DeepInfer kernels, expert parallel, the DSpark confidence head, compact KV truncation, and prefill/decode disaggregation are not implemented.
- DeepGEMM is not linked. `quantizeCols` keeps one scale per weight column for the whole K. A 128-channel scale is a different recipe and is not called from `vgemmBlock`. `MAX_INT32_K` is 131071.

## The pieces

| File | Role |
|---|---|
| `server.js` | WebSocket **signaling** + static hosting. Never sees a weight, an activation, a prompt, or a token. |
| `public/safetensors.js` | header parsing, dtype widening, byte-range coalescing. |
| `public/hf.js` | the Hub client — range reads, gated-repo handling, token redaction. |
| `public/arch.js` | `config.json` -> a normalized spec, and the tensor names to fetch. |
| `public/tokenizer.js` | byte-level BPE, read from the repo's own `tokenizer.json`. |
| `public/shard.js` | the **plan**, per-stage tensor sets, and weight-layout handling. |
| `public/infer.js` | the forward pass, split into `embed` / `runLayers` / `readout`. |
| `public/wire.js` | the binary protocol and the routing decisions, as pure functions. |
| `public/app.js` | the WebRTC mesh, assignments, the token prompt, and the token loop. |
| `public/verified_core.js` | the **verified INT8 units** — unchanged from DaisyChain-Web. |
| `public/webgpu.js` | the WGSL kernels and their **exact init gates** — unchanged. |
| `public/*.bin` | the units as lookup tables — unchanged. |

The last three being unchanged is deliberate. The arithmetic is copied, not
reimplemented, which is what makes the split checkable against the unsplit path
with `!==` instead of a tolerance.

## Reading a model without downloading it

A safetensors file is `[u64 headerLen][JSON header][tensor bytes]`, and the
header gives every tensor's exact byte range. So loading a model here means
fetching `config.json`, the tokenizer, and the weight *headers* — a few tens of
KB regardless of whether the model is 100 MB or 100 GB. After that, every
tensor's location is known and nothing else has been transferred.

`arch.js` turns the config into a normalized spec:

```
family, layers, hidden, heads, kvHeads, headDim, inter, vocab, maxPos,
norm ('rms'|'ln'), normEps, act ('silu'|'gelu'), gated, rope, ropeTheta,
tie, qkvFused, bias, weightLayout ('in_out'|'out_in')
```

Everything downstream works off that spec and never asks which family a model
came from. Unknown architectures are refused by name — silently treating one
block shape as another produces confident nonsense.

Two details that are silent when wrong, and so are decided by evidence rather
than assumption:

- **Weight layout.** `torch.nn.Linear` stores `(out, in)`; GPT-2's `Conv1D`
  stores `(in, out)`, which is already the `k x n` the GEMM wants. A wrong
  transpose does not throw. The layout comes from the spec and is applied once,
  at load.
- **Bias presence.** Qwen2.5 ships q/k/v biases and SmolLM does not, and
  neither says so in `config.json`. Bias is therefore decided by what the
  weight file actually contains, not by a config flag.

## The plan

Whoever loads the model is the **head**: it owns the embedding table and so
does both ends of the pass. It orders devices itself-first, then by peer id,
and apportions blocks by **measured** capacity — each device times a real GEMM
through its own live kernel and reports GEMMs/sec, the same self-calibrating
idea as `cluster.py`'s `capacity_score`, except what is being balanced is
layers per device rather than batch per device.

Apportionment is largest-remainder, so it is a pure function of the capacity
report: every device derives the same plan from the same inputs, and the plan
never has to be trusted, only compared. A device that ends up with zero blocks
is dropped from the ring unless it is the head — an empty hop is pure latency.
*(added 2026-10-22)* A peer whose kernel probe hash disagrees with the head is removed before that
apportionment (`ringMembers`). The head is always kept.

*(added 2026-10-22)* Before the download, `estimatePlanMs` sums each stage's share of
`estimateLapMs` at that stage's own GEMM rate. A slow stage with a real share
of the layers raises the figure. It is a budget check. Above 120000 ms the
head refuses the run. It is not a latency model.

*(added 2026-10-22)* The assignment may set `pass` to `chunk` or `window`. Stages must not guess
the path from the activation length: a chunk can be exactly `T` rows, and
treating it as a full window would drop the KV the other stages kept.

`unpackAssign` refuses any plan whose stages do not cover every layer exactly
once, and any plan whose stage 0 is not the head. A gap there would not throw
at runtime; it would generate fluent text with a layer missing. The repo id is
validated too — it is interpolated into a Hub URL, so a peer must not be able
to point this device at an arbitrary path.

## Fetching a slice

Each device receives an assignment (model id, revision, spec, plan, its own
layer range) and then fetches **its own tensors itself**, with its own
credentials. Weights never travel between peers, which means:

- no device needs to hold the whole model, not even the head;
- a token is never sent over the wire — each device is prompted separately;
- adjacent byte ranges are coalesced, so a stage costs roughly one request
  rather than nine per layer;
- *(added 2026-10-22)* the next range is requested while the previous body is widened to f32.
  The bytes do not change.
- *(added 2026-10-23)* after the slice is arranged, a stage with two or more layers copies those layer weights into one flat buffer and drops the per-layer arrays. Embeddings, the final norm, and `lm_head` stay as separate arrays. A forward gathers at most two layers as views into the flat buffer (`forward_prefetch`, `limit_all_gathers` of 2, reshard after the layer). The views alias the buffer, so the GEMM sees the same bytes. ZeRO stage 1, ZeRO stage 2, a cross-peer all-gather, and tiled re-quantization are not done.
- *(added 2026-10-23)* `layerForward` reads `ly.swa`. A positive window limits the softmax, not the score GEMM. The next queued prompt is tokenized during the wait. JSON masking happens in `sampleAt` after logits exist.

## One token

1. The **head** embeds the last `T` tokens (plus learned positions, for GPT-2)
   and runs its own layers.
2. It sends the hidden state to stage 1. Each stage runs its layers and passes
   the result on.
3. The **last** stage sends the state back to the head, addressed `RETURN`.
4. The head applies the final norm and `lm_head` — for the last position only,
   which at a 150k-token vocabulary turns the largest GEMM in the model into a
   single row — picks a token, and starts the next lap.

The head is both stage 0 and the terminus, so the return leg has its own
address rather than a stage index. Reading it as "stage 0, your turn" was a
real bug: the head re-ran its own layers and the lap never closed.

*(added 2026-10-22)* Two other ways a lap can start:

- **Full window, with a draft.** If a suffix of the token ids already occurred
  earlier, the head appends the tokens that followed that copy (at most 4) and
  reads logits at each of those rows plus one. Matches are kept. The first
  disagreement is kept too. An empty draft is one new token, the same window
  as before. `runLayers` still sees the whole window, so this path stays on
  the bit-exact GEMM.
- **Chunked prefill.** Prompt longer than 32 tokens, and more than one stage.
  Even chunks of about 16 tokens, two in flight. A short tail is folded into
  those chunks so it is not its own hop. `runLayersChunk` appends K and V on the stage
  that owns those layers and attends only the new rows. Decode after that is
  one new token per lap on the same cache. Per-column int8 scales move as the
  cache grows, so this is not bit-identical to the full-window GEMM. A split
  of the chunked path matches itself. `tokenIdx` is the chunk index, and a
  stage applies chunks in order even if a later chunk arrives first.

A second prompt while a run is in progress is queued. If the activation wait
fails or a peer channel closes, the head drops that peer, rebuilds the plan,
and reruns the prompt once. A non-finite hidden state throws inside
`runLayers` / `runLayersChunk` and is not forwarded.

## The wire

All messages are binary. Sentinels continue DaisyChain-Web's numbering (it uses
−2…−8) so a client pointed at the wrong server sees an unknown tag rather than
a valid-looking message of the wrong kind.

| Sentinel | Message |
|---|---|
| `-5` | fragment — large messages chunked at 48 KB |
| `-20` | hello: measured capacity, kernel probe hash, backend |
| `-21` | assignment — model id + your layer range (**no weights**) |
| `-22` | ready — this stage has fetched and loaded its layers |
| `-23` | activation hop |
| `-24` | token emitted (so every device can watch the output) |
| `-25` | run finished |

Activation format:
`[i32 ACT][i32 seq][i32 tokenIdx][i32 nextIndex][u32 actHash][u32 modelHash][f32 hidden]`

The two hashes answer different questions. `actHash` asks whether the payload
arrived intact. `modelHash` asks whether this activation belongs to the model I
hold a slice of — not hypothetical, since reloading the head with a different
model leaves stages out there still holding the old one. It is derived from the
repo id, revision and tensor index rather than the weights, because no device
reads all of them.

## Backends

Unchanged from DaisyChain-Web: DP4A hardware int8 dot → LUT compute shader →
CPU mirror, each **exact-gated at init** and demoted on any mismatch. The CPU
mirror is not an approximation — it produces the same bits, which is what lets
a phone on CPU and a desktop on WebGPU sit in the same ring.

Attention runs per head through the same verified block GEMM, because real
models have head geometries (grouped-query, non-uniform head dims) that the
trainer's fused uniform-head kernels do not cover. Every product still goes
through the units.

There is no plain-float path. If the units fail to load, the device does not
compute.

## What the pipeline changes about verification

In data-parallel training every peer computes the same thing, so broken
arithmetic surfaces as a diverging replica. In a pipeline each stage computes
something different and nobody repeats it — there is no replica to compare
against, and a wrong middle stage yields a confident wrong answer.

What still works: the **kernel probe** (same seeded GEMM everywhere, so it is
comparable even when the real work is not), the init gates, and the live-shape
audit — all of which run per device regardless of what that device is computing.

What is new: activation hashes, the model fingerprint on every hop, and the
**differential check** — the head fetching the whole model and re-running the
prompt locally, then comparing token ids. It is the only instrument that can
prove a distributed answer right rather than merely self-consistent, and it is
offered rather than assumed because it needs a model one device can hold.
