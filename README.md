---
license: mit
tags:
  - distributed-inference
  - pipeline-parallel
  - webgpu
  - webrtc
  - int8
  - old-hardware
---

# ❄ DaisyChain-Infer — run one model across several devices

**Part of DaisyChain on 🤗 Hugging Face → https://huggingface.co/DaisyChainAI**
**Live Space:** https://huggingface.co/spaces/Quazim0t0/DaisyChain-Infer

---

> **In plain terms:** point it at a Hugging Face model, open the page on two or
> more devices, and each one holds only a **slice of the model's layers**. A
> token is produced by passing the hidden state around the ring, peer-to-peer.
> The group can run a model that **no single device could hold**.
> Before you rely on it, read [Honest limits](#honest-limits).

This repo is the **project and the guide**. Clone it and run it on your own
machines:

```bash
npm install
npm start            # http://localhost:8788
npm test             # 47 checks: pipeline equivalence, wire protocol, loader
```

Full walkthrough: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** ·
Internals: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** ·
The guide as one page: **[GUIDE.md](GUIDE.md)**

---

## What this adds to DaisyChain

DaisyChain-Train is explicit about its limit: it **pools compute, not memory**.
Every node holds a full replica, so a model bigger than one machine cannot be
trained, and chaining five laptops does not give you one big machine.

Inference is where that limit can be lifted, because a forward pass is a chain:
layer `l` needs layer `l-1`'s **output**, never its **weights**. So the layers
live on different machines and the activation travels instead.

And because safetensors gives every tensor an exact byte range, **each device
fetches only its own layers straight from the Hub**. No device — not even the
one driving the run — ever holds the whole model. That is what makes the pooling
real rather than a redistribution of something one machine already had to load.

|  | DaisyChain-Train | DaisyChain-Infer |
|---|---|---|
| What is split | the **batch** | the **model** |
| What crosses the wire | gradients (whole-model sized, every step) | hidden states (`T×hidden` floats, per hop) |
| Every device holds | the entire model | its own layers only |
| Pools | compute | **memory** |
| More devices means | more throughput | a **bigger model fits** |

Running SmolLM-135M across three devices, each downloads and holds 135–243 MB
of a 513 MB model, and the activation moving between them is tens of KB. That
asymmetry is why this works.

## The ring

```
        ┌──────────────── hidden state (T×hidden f32) ─────────────┐
        ▼                                                          │
   ┌─────────┐        ┌─────────┐        ┌─────────┐               │
   │ stage 0 │───────▶│ stage 1 │───────▶│ stage 2 │───────────────┘
   │  HEAD   │        │ layers  │        │ layers  │
   │ embed   │        │ 10–19   │        │ 20–29   │
   │ layers  │        └─────────┘        └─────────┘
   │  0–9    │
   │ lm_head │◀── the returning state becomes the next token
   └─────────┘
```

It is a **ring, not a line**, and weight tying forces that. When `lm_head` is
tied to the embedding table — as it is in most small models — the largest tensor
is needed at *both* ends: to embed the prompt and to produce the logits. Copying
it to the last stage would hand back most of the memory just pooled, so the last
stage returns its hidden state to the head, which owns the embedding **once**.

A consequence worth stating plainly: a middle stage receives layer weights only.
It never gets the embedding table, so **it never sees the vocabulary** — it
passes floats it cannot interpret.

## Models it can run

Any Hugging Face repo with `.safetensors` weights whose architecture is:

- **Llama-style** — Llama, Mistral, Qwen2/2.5, SmolLM, TinyLlama
  (RMSNorm, RoPE, grouped-query attention, SwiGLU)
- **GPT-2-style** — LayerNorm with bias, learned positions, fused QKV, GELU

Verified end to end: `HuggingFaceTB/SmolLM-135M`, `openai-community/gpt2`,
`Qwen/Qwen2.5-0.5B`.

Anything else is **refused by name**, not approximated. Treating an unknown
architecture as a known one produces fluent, confident, wrong output — the
failure this project spends its verification budget making impossible. The same
applies to tokenizers: byte-level BPE is implemented, and a SentencePiece or
WordPiece repo is rejected rather than tokenized approximately.

Two details that are silent when wrong, and so are decided by evidence rather
than assumption:

- **Weight layout.** `torch.nn.Linear` stores `(out, in)`; GPT-2's `Conv1D`
  stores `(in, out)`, which is already the `k×n` the GEMM wants. A wrong
  transpose does not throw.
- **Bias presence.** Qwen2.5 ships q/k/v biases and SmolLM does not, and neither
  says so in `config.json`. Bias is decided by what the weight file contains.

## Credentials

Public models need nothing. For gated or private ones, the **local** build asks
for a read token once, when a request actually fails for want of it — held in
memory for that tab only, never written to storage, never logged, never put in a
URL, and **never sent to another device** (each device authenticates itself,
because each fetches its own layers).

The hosted Space instead offers **Sign in with Hugging Face** and has no
paste-a-token box at all, because typing a personal access token into a page you
do not control is a bad habit even when the code is honest.

## Verification

The trainer's stack carries over unchanged: exact init gates on every kernel on
every boot, the continuous random-cell audit at live shapes, and the cross-device
kernel probe.

But a pipeline moves the risk somewhere those instruments cannot reach:

> In data-parallel **training**, every peer computes the same thing, so a device
> with broken arithmetic shows up as a diverging replica. In a **pipeline**, each
> stage computes something *different* and nobody else repeats it. There is no
> replica to compare against. A wrong middle stage produces a fluent, confident,
> wrong answer, and no consistency check anywhere in the system would notice.

Four things close that:

1. **The kernel probe**, which matters more here than in the trainer — the same
   seeded GEMM on every device, so it stays comparable even when the real work is
   not. A stage whose probe disagrees is flagged before it is given layers.
2. **Activation integrity hashes** on every hop, plus the model fingerprint, so a
   stage still holding a slice of a *different* model refuses the hop instead of
   silently mixing two models.
3. **Structural validation** of every assignment — a plan that does not cover
   each layer exactly once is refused, because it would still generate fluent
   text with a layer missing.
4. **The differential check** — the head re-running the identical prompt with
   every layer locally and comparing token ids. The only instrument that can show
   a distributed answer is *right* rather than merely self-consistent.

`test_pipeline.js` asserts that splitting changes **no bit**, for both
architecture families, across several uneven splits including a head that keeps
no layers. Two cases are mutation checks — a stage that silently drops a layer,
and stages applied out of order — both of which *must* fail the comparison.
Without those, a test where both sides call the same code proves nothing.

Confirmed on a real model too: SmolLM-135M's 30 layers split `[10,10,10]`,
`[1,14,15]`, `[0,15,15]` and `[5,5,5,5,5,5]` all produced token sequences
identical to the single-device run.

`test_loader.js` builds safetensors files and reads them back, checks BF16/F16
widening is exact (including subnormals and signed zero), and round-trips the BPE
tokenizer. `test_wire.js` round-trips the protocol and refuses the malformed
messages that would otherwise produce plausible wrong answers — a plan with a
gap, a crafted repo id, a one-ulp flip deep in an activation, a `-0` flipped to
`+0`.

### One bug, and what it cost to find

The first live two-device run stalled. The head is both stage 0 *and* the ring's
terminus, so an activation addressed to index 0 meant "start the lap" outbound
and "the lap is finished" inbound. The head read the return leg as its own turn,
re-ran its own layers, forwarded again, and the lap never closed.

Every number was correct. Every message round-tripped. **Neither test suite could
see it** — the pipeline test calls the stages in order itself, and the codec test
only checks bytes. The defect lived in the *route*, precisely the category
DaisyChain-Web's own self-corpus writeup identified as needing different
instruments rather than better oracles. The fix gave the return leg its own
address, and the routing decision moved out of the event handler into a pure
function so `walkLap` in `test_wire.js` can walk a lap and assert it visits each
stage once and terminates. That test fails against the old behaviour; it was
checked.

## Layout

```
server.js                 signaling + static host + OAuth code exchange
public/safetensors.js     header parsing, exact BF16/F16 widening, range coalescing
public/hf.js              Hub client: range reads, gated repos, token redaction
public/arch.js            config.json -> normalized spec; per-family tensor names
public/tokenizer.js       byte-level BPE from the repo's own tokenizer.json
public/shard.js           the plan, per-stage tensor sets, weight-layout handling
public/infer.js           forward pass: embed / runLayers / readout
public/wire.js            binary protocol + routing, as pure functions
public/app.js             WebRTC mesh, assignments, credentials, token loop
public/verified_core.js   the verified INT8 units      (unchanged from DaisyChain-Web)
public/webgpu.js          WGSL kernels + exact gates   (unchanged)
public/*.bin              the units as lookup tables   (unchanged)
test_pipeline.js          split vs unsplit, bit-exact, both families
test_wire.js              protocol round-trips, refusals, ring routing
test_loader.js            safetensors, dtype widening, BPE
docs/                     GETTING_STARTED, ARCHITECTURE
```

## Honest limits

- **Latency, not bandwidth, is the cost.** Every token pays one round trip per
  stage. More stages buy capacity, not speed — expect tokens/sec to *fall* as you
  add devices.
- **No KV cache.** Every token re-runs the whole window, which is also what keeps
  the split bit-comparable against an unsplit run.
- **f32 in memory.** Weights are widened from F16/BF16 on load, so a stage costs
  4 bytes per parameter it holds.
- **The head is a single point of failure**, and a stage that drops stalls the
  ring; press Generate again to re-plan around whoever is still connected.
- **No authentication of activations.** A malicious stage that runs correct math
  but returns a crafted activation is not caught by any of this. Peers see each
  other's IPs, and the head sees your prompt. Ring with people you trust.
- **Proof of concept**, not a hardened service.

---

**License:** MIT · **Author:** Dean Byrne (Quazim0t0) · **Org:** DaisyChainAI

Built on [DaisyChain-Train](https://huggingface.co/DaisyChainAI/DaisyChain-Train)
and [DaisyChain-Web](https://huggingface.co/spaces/Quazim0t0/DaisyChain-Web).

## Citation

```bibtex
@misc{byrne2026daisychaininfer,
  title        = {DaisyChain-Infer: Layer-Sharded Peer-to-Peer Inference in the Browser},
  author       = {Byrne, Dean (Quazim0t0)},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/DaisyChainAI/DaisyChain-Infer}},
  note         = {Run one model across several devices; pools memory, not just compute}
}
```
