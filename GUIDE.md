# ❄ DaisyChain-Infer — run one model across your devices

Part of **DaisyChain** → https://huggingface.co/DaisyChainAI

Point it at a **Hugging Face model**, open the page on two or more devices, and
each one takes a slice of the model's layers. A token is produced by passing
the hidden state around the ring, peer-to-peer over WebRTC. Every multiply runs
through the same **verified INT8 units** as the rest of DaisyChain — WebGPU
where available, the identical math on CPU where not.

**This runs locally.** It is a project you clone and start on your own
machines; there is no hosted deployment, and it is not a Space.

## Added 2026-10-22

The rest of this file is the previous guide. This list, and every later note marked *(added 2026-10-22)*, is new.

- Compute cores from DaisyChain-Train `a28437c`. Non-finite activations are refused.
- A disagreeing kernel probe is left out of the plan.
- Lap budget is the plan, each stage at its own rate. Above 120 seconds per token the run stops before download.
- A second prompt waits.
- If the browser reports enough RAM for the whole model, the turn stays on the head.
- A 64-token prefix hash is kept per conversation and read back on load. The full-window pass still recomputes it.
- Long multi-device prompts use even chunks of about 16 tokens, two in flight, with per-stage KV. Verify reruns that path.
- A repeated suffix is drafted and scored in one lap on the full-window path.
- One weight range stays in flight while the previous body is widened to f32.
- A dropped peer or a stalled lap is replayed once. Weights still come from the Hub.

## Added 2026-10-23

- A stage with two or more layers packs those weights into one flat buffer and unpacks at most two layers during a forward. Embeddings and `lm_head` stay put. Weights do not cross the ring.
- Optimizer states, reduced gradients, a peer parameter all-gather, and tiled re-quantization are not implemented.
- From DeepInfer: sliding-window softmax mask, sampling tied to the token position, a JSON mask on the top 40 completion tokens, and tokenizing the next queued prompt during the current turn.
- Not from DeepInfer: CUDA kernels, expert parallelism, a confidence-head draft model, compact KV replay, disaggregated prefill.
- DeepGEMM's FP8 and FP4 kernels are not used. A scale every 128 channels of K is a different number from the one INT8 scale per row and per column.

```bash
npm install
npm start            # http://localhost:8788
```

Built from [DaisyChain-Train](https://huggingface.co/DaisyChainAI/DaisyChain-Train)
and [DaisyChain-Web](https://huggingface.co/spaces/Quazim0t0/DaisyChain-Web) —
the verified units, the WGSL kernels and their exact gates, and the WebRTC mesh
are carried over unchanged.

---

## What is actually new here

DaisyChain-Train is explicit about its limit: it **pools compute, not memory**.
Every node holds a full replica, so a model bigger than one machine cannot be
trained, and chaining five laptops does not give you one big machine.

Inference is where that limit can be lifted, because a forward pass is a chain:
layer `l` needs layer `l-1`'s **output**, never its **weights**.

And because safetensors gives every tensor an exact byte range, **each device
downloads only its own layers, straight from the Hub**. No device — not even
the one driving the run — ever holds the whole model.

|  | DaisyChain-Train | DaisyChain-Infer |
|---|---|---|
| What is split | the **batch** | the **model** |
| What crosses the wire | gradients (whole-model sized, every step) | hidden states (`T×hidden` floats, per hop) |
| Every device holds | the entire model | its own layers only |
| Pools | compute | **memory** |
| More devices means | more throughput | a **bigger model fits** |

Running SmolLM-135M across three devices, each one downloads and holds
135–243 MB of a 513 MB model. The activation moving between them is tens of KB.
That asymmetry is why this works.

## Scheduling *(added 2026-10-22)*

The compute units are the live DaisyChain-Train browser cores (Hub revision `a28437c`, 20 October 2026). A non-finite activation throws `NonFiniteError` and is not forwarded.

- A peer whose kernel probe disagrees is left out of the plan.
- Layers are still split by measured GEMMs/sec.
- Above **120 seconds** per estimated token, the run stops before any weight download.
- A second prompt waits. The turn already generating keeps the ring.
- If the browser reports enough RAM for the whole model, the turn stays on the head. Unknown RAM uses the ring.
- A **64-token** prefix hash is recorded per conversation, keyed by the first block, so two chats do not erase each other. The full-window pass still recomputes it.
- A prompt longer than **32 tokens** on more than one device is prefetched in even chunks of about **16** tokens, two in flight, with per-stage KV. That path is not the full-window GEMM. Verify reruns the chunked path.
- On the full-window path, a repeated suffix is drafted and scored in one lap. Matching drafts are kept, plus the first disagreement.
- One weight range stays in flight while the previous body is widened to f32.
- A dropped peer or a stalled lap is replayed **once** from the start. Weights still come from the Hub.

## Parameter shard *(added 2026-10-23)*

ZeRO stages 1 and 2 do not apply here. There is no optimizer and no backward. Parameters are not gathered from other devices.

When a stage owns two or more layers, those weights sit in one flat buffer. A forward unpacks the current layer and the next one, runs the layer, then drops the one just used. At most **2** layers are unpacked. A single layer stays resident. The embedding and `lm_head` are not in the shard. The bytes the GEMM reads are the same bytes, so a full-window split stays bit-identical. The flat buffer does not reduce the stage below one copy of its parameters.

## DeepInfer *(added 2026-10-23)*

DeepInfer's CUDA kernels, expert parallelism, and KV cluster are not used.

- A model with `use_sliding_window` and a positive `sliding_window` masks attention outside that window on layers at or after `max_window_layers`. The GEMM still includes every key, so the int8 scales do not change. Compact replay, which drops old keys, is not done.
- Above temperature 0, each token position has its own draw from the seed.
- Output set to JSON keeps the completion inside a JSON value, using the top **40** tokens. The hidden state is already finished before the mask.
- The next queued prompt is tokenized while the current turn runs. One turn still owns the ring.

## DeepGEMM *(added 2026-10-23)*

DeepGEMM's CUDA GEMMs, grouped expert GEMMs, and indexer are not used. The verified GEMM keeps one INT8 scale for a whole activation row and one for a whole weight column. A 128-channel scale would change the products. K above **131071** is still refused.

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
tied to the embedding table — as it is in most small models — the largest
tensor is needed at *both* ends: to embed the prompt and to produce the logits.
Copying it to the last stage would hand back most of the memory just pooled, so
the last stage returns its hidden state to the head, which owns the embedding
**once** and does both ends.

A consequence worth stating plainly: a middle stage receives layer weights
only. It never gets the embedding table, so **it never sees the vocabulary** —
it passes floats it cannot interpret.

## Models it can run

Any Hugging Face repo with `.safetensors` weights whose architecture is:

- **Llama-style** — Llama, Mistral, Qwen2/2.5, SmolLM, TinyLlama
  (RMSNorm, RoPE, grouped-query attention, SwiGLU)
- **GPT-2-style** — LayerNorm with bias, learned positions, fused QKV, GELU

Verified working end to end: `HuggingFaceTB/SmolLM-135M`,
`openai-community/gpt2`, `Qwen/Qwen2.5-0.5B`.

Anything else is **refused by name**, not approximated. Treating an unknown
architecture as a known one produces fluent, confident, wrong output, which is
the failure this project spends its verification budget making impossible. The
same applies to tokenizers: byte-level BPE is implemented, and a
SentencePiece/Unigram repo is rejected rather than tokenized approximately.

Practical ceiling: weights are held as f32, so budget ~4 bytes per parameter
across the group, and remember the head also carries the embedding table.

## The token

Gated or private models need a Hugging Face token. You are asked for one
**once**, when a request actually fails for want of it — not up front.

It is held in memory for that tab and nowhere else: **not** localStorage, not
sessionStorage, not a cookie, not the URL, never written to the log, and
**never sent to another device**. Each device is asked for its own, because
each device fetches its own layers. Reloading the tab forgets it, and there is
a *Forget token* button.

## Verification

The trainer's stack carries over unchanged: exact init gates on every kernel on
every boot, the continuous random-cell audit at live shapes, and the
cross-device kernel probe.

But a pipeline moves the risk somewhere those instruments cannot reach:

> In data-parallel **training**, every peer computes the same thing, so a
> device with broken arithmetic shows up as a diverging replica. In a
> **pipeline**, each stage computes something *different* and nobody else
> repeats it. There is no replica to compare against. A wrong middle stage
> produces a fluent, confident, wrong answer, and no consistency check
> anywhere in the system would notice.

Four things close that:

1. **The kernel probe**, which matters more here than in the trainer — the same
   seeded GEMM on every device, so it stays comparable even when the real work
   is not. A stage whose probe disagrees is left out of the plan. *(added 2026-10-22)*
2. **Activation integrity hashes** on every hop, plus the model fingerprint, so
   a stage still holding a slice of a *different* model refuses the hop instead
   of silently mixing two models.
3. **Structural validation** of every assignment — a plan that does not cover
   each layer exactly once is refused, because it would still generate fluent
   text with a layer missing.
4. **The differential check** (tick *Verify*). The driving device re-runs the
   identical prompt with every layer locally and compares token ids. It is the
   only instrument that can show a distributed answer is *right* rather than
   merely self-consistent.

```bash
npm test        # pipeline equivalence + wire protocol + loader
```

`test_pipeline.js` asserts that splitting changes **no bit**, for *both*
architecture families, across several uneven splits including a head that keeps
no layers. Two cases are mutation checks — a stage that silently drops a layer,
and stages applied out of order — both of which *must* fail the comparison.
Without those, a test where both sides call the same code proves nothing.
*(added 2026-10-22)* The same file also checks a dropped probe, a refused non-finite activation,
a chunked prefill across a split, and a lookahead draft across a split. The
chunked path is compared to itself, not to the full-window GEMM.
*(added 2026-10-23)* The same file checks that a flat parameter shard matches a resident forward, on the full window and on one chunk, and that a third live layer is refused. It also checks a sliding-window mask across a split, a JSON mask, a position seed, a prepared prompt, and that a 128-channel weight scale differs from the verified column scale.

This was also confirmed against a real model: SmolLM-135M's 30 layers split
`[10,10,10]`, `[1,14,15]`, `[0,15,15]` and `[5,5,5,5,5,5]` all produced token
sequences identical to the single-device run.

`test_loader.js` builds safetensors files and reads them back, checks BF16/F16
widening is exact (including subnormals and signed zero), and round-trips the
BPE tokenizer. `test_wire.js` round-trips the protocol and refuses the
malformed messages that would otherwise produce plausible wrong answers — a
plan with a gap, a crafted repo id, a one-ulp flip deep in an activation, a
`-0` flipped to `+0`.

### One bug, and what it cost to find

The first live two-device run stalled. The head is both stage 0 *and* the
ring's terminus, so an activation addressed to index 0 meant "start the lap"
outbound and "the lap is finished" inbound. The head read the return leg as its
own turn, re-ran its own layers, forwarded again, and the lap never closed.

Every number was correct. Every message round-tripped. **Neither test suite
could see it** — the pipeline test calls the stages in order itself, and the
codec test only checks bytes. The defect lived in the *route*, precisely the
category DaisyChain-Web's own self-corpus writeup identified as needing
different instruments rather than better oracles. The fix gave the return leg
its own address, and the routing decision moved out of the event handler into a
pure function so `walkLap` in `test_wire.js` can walk a lap and assert it
visits each stage once and terminates. That test fails against the old
behaviour; it was checked.

## Honest limits

- **Latency, not bandwidth, is the cost.** Every token pays one round trip per
  stage. More stages buy capacity, not speed — expect tokens/sec to *fall* as
  you add devices.
- **The full-window path has no KV cache.** *(added 2026-10-22, replaces "No KV cache".)* Every token re-runs the window, which
  is what keeps that path bit-comparable against an unsplit run. Context length
  is the dominant per-token cost. A long multi-device prompt uses chunked
  prefill instead, and that is a different number.
- **f32 in memory.** Weights are widened from F16/BF16 on load, so a stage
  costs 4 bytes per parameter it holds.
- **The head is a single point of failure.** *(added 2026-10-22.)* A stage that drops is removed and the
  prompt is replayed once. If that also fails, press Generate again.
- **No authentication of activations.** A malicious stage that runs correct
  math but returns a crafted activation is not caught by any of this — the
  verification proves the *computation* is right on every honest device. Peers
  see each other's IPs, and the head sees your prompt. Run rings with people
  you trust.

---

**License:** MIT · **Author:** Dean Byrne (Quazim0t0) · **Org:** DaisyChainAI
