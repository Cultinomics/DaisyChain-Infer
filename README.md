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

## Added 2026-10-22

The rest of this file is the previous README. Everything in this list, and every later note marked *(added 2026-10-22)*, is new.

- Compute cores copied from live DaisyChain-Train, Hub revision `a28437c` (20 October 2026). Lookup tables unchanged.
- A non-finite activation throws `NonFiniteError` and is not forwarded. The same cores refuse an int32 accumulator that would wrap, and a non-finite gradient before DaisyAdam.
- A peer whose kernel probe disagrees is left out of the plan.
- Layers stay split by measured GEMMs/sec, so a slower device gets fewer layers.
- Lap budget uses `estimatePlanMs`: each stage's share of the model, at that stage's own rate, after the plan is final. Above 120 seconds per token the run stops before any download.
- A second prompt is queued. The turn already generating keeps the ring.
- If `navigator.deviceMemory` says the whole model fits in a quarter of that RAM, the turn stays on the head. Unknown memory uses the ring.
- A 64-token prefix hash is kept per conversation, keyed by the first block, written to IndexedDB, and read back when the page loads. Two chats do not wipe each other. The full-window pass still recomputes a matched prefix. Oldest record drops at 32 MB.
- Prompt longer than 32 tokens, more than one device: even chunks of about 16 tokens, two in flight, per-stage KV. Not bit-identical to the full-window GEMM. Verify reruns this path. The assignment field `pass` is `chunk` or `window`. Stages do not guess from the activation length.
- Full-window path: a repeated suffix is drafted and scored in one lap (`lookaheadFrame`). Matches are kept, plus the first disagreement.
- One weight range stays in flight while the previous body is widened to f32.
- A planned peer other than the head dropping, or a stalled lap, replays the prompt once. Weights still come from the Hub.

## Added 2026-10-23

- ZeRO and FSDP, forward only, inside one stage. A stage with two or more layers packs those weights into one flat buffer and unpacks at most two layers for the forward. Embeddings and `lm_head` stay as they are. Weights still come from the Hub and do not cross the ring.
- ZeRO stages 1 and 2 are not implemented. Infer has no optimizer and no backward. Parameters are not all-gathered across peers. `TiledLinear` is not used.
- From DeepInfer: a sliding-window mask on layers that configure one, sampling tied to the token position, a JSON mask on the top 40 completion tokens, and tokenizing the next queued prompt while the current turn still runs.
- Not taken from DeepInfer: CUDA kernels, expert parallelism, a draft-model confidence head, compact KV replay, disaggregated prefill, and a KV store.
- DeepGEMM is not a second GEMM. Its FP8 and FP4 kernels, 128-channel scales, grouped MoE GEMM, and indexer stay out. The verified path keeps one INT8 scale per activation row and one per weight column.

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
npm test             # 59 checks: pipeline equivalence, wire protocol, loader
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

## Scheduling *(added 2026-10-22)*

The compute units are the live DaisyChain-Train browser cores (Hub revision `a28437c`, 20 October 2026), not the older snapshot this repo shipped with. A non-finite activation throws `NonFiniteError` before any layer runs, and the next stage does not forward it. The same cores refuse an int32 accumulator that would wrap, and they refuse a non-finite gradient before DaisyAdam. The lookup tables did not change.

On top of that, the head does a few things the old loop did not:

- A peer whose kernel probe disagrees is left out of the plan, not only warned.
- Layers are still split by measured GEMMs/sec, so a slower device gets fewer layers.
- Before any weight download, `estimatePlanMs` sums each stage's share at that stage's own rate. Above **120 seconds** per token the run stops. *(added 2026-10-22)*
- A second prompt while a turn is generating is queued. The turn already on the ring finishes first.
- If `navigator.deviceMemory` says the whole model fits in a quarter of that RAM, the turn runs on the head and the ring is not used. Unknown memory uses the ring. A split that the head can hold only adds a hop.
- A **64-token** block hash is kept per conversation, keyed by the first block, in memory and in IndexedDB when the browser has it. Two chats do not wipe each other's record. The oldest record is dropped at **32 MB**. The full-window pass still recomputes. A prefix hit is logged and then computed again, because reusing K and V would change the per-column int8 scales.
- With more than one device and a prompt longer than **32 tokens**, prefill goes out in even chunks of about **16** tokens, two chunks in flight. A short tail is not left as its own hop. Each stage keeps K and V for its own layers. Decode stays one new token per lap. This path is not bit-identical to one full-window GEMM. Verify reruns the same chunked path.
- On the full-window path, a repeated suffix proposes the tokens that followed the earlier copy (prompt lookup, no second model). One lap scores that span. Matching drafts are kept, plus the first token that disagreed. If nothing repeats, the lap still produces one token.
- Weight fetches keep one HTTP range in flight while the previous body is widened to f32. The bytes are the same.
- If a peer disconnects or a lap stalls, the head drops that peer, replans, and reruns the prompt **once** from the start. Weights still come from the Hub, not from another device.

## Parameter shard *(added 2026-10-23)*

ZeRO and FSDP split training state across data-parallel ranks. This project does not train, so the parts that move optimizer moments or gradients are left out.

What a stage does when it owns two or more layers:

- Those layers' float32 weights are copied once into one flat buffer. The per-layer arrays are then dropped. The embedding, the final norm, and `lm_head` stay where they are. `lm_head` stays on the head.
- A forward gathers the current layer as views into that buffer, prefetches the next layer, runs the same layer, then releases the one just used. At most **2** layers are unpacked.
- The views alias the flat buffer. The GEMM reads them. The full-window path stays bit-identical to keeping every layer resident. The chunked path uses the same gather.
- A stage with one layer keeps that layer resident.
- Peak RAM is still one copy of the stage's parameters. The flat buffer does not make the slice smaller.
- Weights are not sent to another device. Each device still fetches its own slice from the Hub.

Left out on purpose:

- ZeRO stage 1 (optimizer states) and stage 2 (reduced gradients). There is no Adam state and no backward.
- A parameter all-gather across the ring. That would move weights on every token.
- `TiledLinear`. Re-quantizing a matrix per tile would change the int8 scales.
- NVMe offload. The flat buffer is the CPU-resident copy.

## DeepInfer *(added 2026-10-23)*

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) serves DeepSeek-V4.1-Flash on NVIDIA SM100 and Ascend. Most of that stack does not fit a browser ring that only runs Llama-style and GPT-2-style blocks.

Taken:

- **Sliding-window mask.** If `use_sliding_window` is set and `sliding_window` is positive, layers with index `>= max_window_layers` drop attention outside that window. The GEMM still sees every key column, so the int8 scales stay the scales of the full matrix. A window wider than the cache matches full causal attention. `use_sliding_window: false` leaves the mask off.
- **Position seed.** Above temperature 0, the draw for token position `p` depends on the seed and `p`. Scoring a longer draft does not move a later draw.
- **JSON mask.** The Output control can require the completion to stay inside a JSON value. The head checks the top **40** tokens after the hidden state is already computed. No legal token in that set falls back to the unconstrained pick. This does not change the hop.
- **Queued prompt.** The next prompt is tokenized while the current turn still holds the ring. The forward stays one turn at a time.

Left out:

- FlashMLA, DeepGEMM, DeepEP, TileKernels, and the other CUDA kernels. The verified INT8 units stay.
- Attention data parallel, context parallel, MoE expert parallel, and expert load balancing.
- DSpark's confidence head and next-n draft model. The prompt-lookup draft stays capped at **4** and is scored in the same window. Shortening that draft would not shrink the GEMM.
- Compact SWA replay (the V4.1 keep of **140**, or a **128**-token recompute). Dropping key rows would change int8 scales. Prefix hits still recompute.
- Disaggregated prefill and decode, the 3FS KV store, FP4 weights, and the multi-protocol recipe tokenizer.

## DeepGEMM *(added 2026-10-23)*

[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) is a CUDA kernel library for SM90 and SM100. It compiles FP8, FP4, and BF16 GEMMs at runtime, plus grouped MoE GEMMs, a fused Mega MoE, and the V3.2 indexer. None of those kernels run in this browser ring.

The numeric recipe is the part that would matter, and it does not match. DeepGEMM's FP8 GEMM uses one scale per **128** channels of K (`BLOCK_K == 128`). A spike in a later block does not widen the scale of an earlier block. `vgemmBlock` uses one INT8 scale for the whole activation row and one for the whole weight column. Swapping in the 128-channel scale would change every product, including the split against the unsplit run. The test `a 128-channel weight scale is not the verified column scale` pins that difference. The forward path does not call a block scale.

Also left out:

- Grouped GEMMs, masked grouped GEMMs, and the K-grouped backward. This ring has no experts and no backward.
- Mega MoE, NVLink overlap, and the lightning-indexer ReLU logits.
- UE8M0 packed scales, TMA layout, programmatic dependent launch, and JIT CUDA.
- A non-deterministic fast GEMM. The CPU mirror and the GPU kernel stay on the same INT8 units. K above **131071** is refused so the int32 accumulator does not wrap.

Weights are already stored `k x n` once at load. That is the layout a non-transposed A times a transposed B expects. It is not a new kernel.

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
   not. A stage whose probe disagrees is left out of the plan. *(added 2026-10-22)*
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
*(added 2026-10-22)* The same file checks that a disagreeing probe is dropped, that a non-finite
activation is refused, that a chunked prefill matches across a split, and that
a lookahead draft matches across a split. The chunked path is compared to
itself, not to the full-window GEMM. *(added 2026-10-23)* It also checks that a
flat parameter shard matches a resident forward on both the full-window path
and a chunk, that the flat bytes are unchanged, and that a third live layer
is refused. *(added 2026-10-23)* It also checks a sliding-window mask across a
split, a JSON mask, a position seed, and a prepared prompt id list. A
128-channel weight scale is checked to be a different number from the
verified whole-column scale.

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
public/shard.js           the plan, probe filter, lap estimate, prefix index, drafts   (added 2026-10-22); flat parameter shard (added 2026-10-23)
public/infer.js           forward pass, chunked prefill, lookahead readout          (added 2026-10-22); gather window of 2 (added 2026-10-23)
public/wire.js            binary protocol + routing, as pure functions
public/app.js             WebRTC mesh, assignments, queue, replay, token loop       (added 2026-10-22)
public/verified_core.js   the verified INT8 units      (copied from live DaisyChain-Train, 2026-10-22)
public/webgpu.js          WGSL kernels + exact gates   (copied from live DaisyChain-Train, 2026-10-22)
public/*.bin              the units as lookup tables   (unchanged; same as Train)
test_pipeline.js          split vs unsplit, bit-exact, both families
test_wire.js              protocol round-trips, refusals, ring routing
test_loader.js            safetensors, dtype widening, BPE
docs/                     GETTING_STARTED, ARCHITECTURE
```

## Honest limits

- **Latency, not bandwidth, is the cost.** Every token pays one round trip per
  stage. More stages buy capacity, not speed — expect tokens/sec to *fall* as you
  add devices.
- **The full-window path has no KV cache.** *(added 2026-10-22, replaces "No KV cache".)* Every token re-runs the window, which
  is what keeps that path bit-comparable against an unsplit run. A long prompt on
  a multi-device ring uses the chunked path instead, and that path is a different
  number.
- **f32 in memory.** Weights are widened from F16/BF16 on load, so a stage costs
  4 bytes per parameter it holds.
- **The head is a single point of failure.** *(added 2026-10-22.)* A stage that drops is removed and the
  prompt is replayed once. If that also fails, press Generate again.
- **No optimizer shard, and no weight gather across peers.** *(added 2026-10-23.)* A stage with two or more layers keeps one flat copy and unpacks at most two layers. The slice is not smaller than one copy of its parameters. Weights still come from the Hub.
- **No DeepInfer kernel stack.** *(added 2026-10-23.)* No expert parallel, no draft-model head, no compact KV drop, no disaggregated prefill. A sliding window is a softmax mask. JSON is a pick mask.
- **No DeepGEMM kernel.** *(added 2026-10-23.)* FP8, FP4, 128-channel scales, and grouped MoE GEMMs are not used. One INT8 scale covers each activation row and each weight column.
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
