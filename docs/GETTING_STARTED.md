# Getting started with DaisyChain-Infer

This is a local project. There is no hosted version — you clone it and run it
on your own machines.

## Run the server

```bash
npm install
npm start            # http://localhost:8788
```

Open it in two tabs to try the ring on one machine, or on two devices to try it
for real.

> **HTTPS matters.** WebGPU and cross-device WebRTC need a *secure context*:
> `localhost` or HTTPS. Plain `http://192.168.x.x` from another device will not
> get WebGPU and may not connect at all. For real multi-device runs on your own
> network, put it behind a tunnel (`cloudflared`, `ngrok`) or a local TLS
> reverse proxy.

## Pick a model

Type any Hugging Face repo id — `owner/name` — and press **Load**.

Works with `.safetensors` repos whose architecture is:

- **Llama-style**: Llama, Mistral, Qwen2/2.5, SmolLM, TinyLlama
- **GPT-2-style**: GPT-2 and close relatives

Good ones to start with:

| Repo | Size | Notes |
|---|---|---|
| `HuggingFaceTB/SmolLM-135M` | 513 MB f32 | 30 layers — splits nicely |
| `openai-community/gpt2` | 523 MB f32 | the GPT-2 path |
| `Qwen/Qwen2.5-0.5B` | 942 MB f32 | GQA 14/2, and it ships QKV biases |

Loading reads only `config.json`, `tokenizer.json` and the weight **headers** —
a few tens of KB, whatever the model's size. No weights move until you press
Generate, and then each device fetches only its own layers.

Weights are held as **f32**, so budget ~4 bytes per parameter across the group.
The ring plan shows exactly how many MB each device will download and hold.

An unsupported architecture or tokenizer is refused with a message naming what
it found. That is deliberate: approximating one would produce fluent, confident,
wrong text.

## Gated or private models

You are asked for a Hugging Face token **once**, and only when a request
actually fails for want of one. Create a **read** token at
huggingface.co/settings/tokens.

The token is held in memory for that tab and nowhere else — not localStorage,
not sessionStorage, not a cookie, not the URL, never logged, and **never sent
to another device**. Each device is prompted for its own, because each device
downloads its own layers. Reloading the tab forgets it, and *Forget token*
clears it immediately.

For a gated model, accept its licence on the model page first — a token alone
will not get past a licence you have not accepted.

## Rooms

Devices on one network group automatically (by public IP, Snapdrop-style). To
include devices on other networks, everyone opens:

```
https://<host>/?room=MY-SECRET-CODE
```

The first person in is the **host** and approves each device individually.

## Run it

1. Wait until the devices see each other in the peer list.
2. On **one** device, load the model. That device becomes the **head**: it
   holds the embedding table, embeds your prompt, and turns the returning
   hidden state back into words.
3. Check the **ring plan** — who holds which layers, and how much each will
   download.
4. Type a prompt and press **Generate**. Each device fetches its layers (this
   takes a moment the first time), reports ready, and the ring starts.

| Setting | Default | What it does |
|---|---|---|
| Tokens | 60 | how many to generate |
| Context length | 64 | window size; the main cost per token (there is no KV cache) |
| Temperature ÷100 | 0 | 0 = greedy, deterministic and therefore checkable |
| Seed | 1234 | drives sampling above temperature 0, so runs stay reproducible |
| Verify | off | afterwards, re-run with every layer here and compare |

Every device shows the token stream as it arrives — including ones holding a
few layers and no vocabulary.

## Reading the numbers

- **tokens/sec** — the whole ring's rate. Expect it to *fall* as you add
  stages: each one adds a round trip per token. More stages buy capacity, not
  speed.
- **ring hops** — stages × tokens.
- **kernel probe** (in the log) — the same number on every honest device,
  whatever its backend. A peer reporting a different one is computing different
  arithmetic, and the log says so.
- **my slice** — which layers this device holds and how much memory they take.

## Verify

Tick **Verify** before generating. Afterwards the head downloads the whole
model and re-runs the identical prompt locally, then compares token ids:

```
VERIFIED: the distributed run and the single-device run produced identical token ids.
```

In a pipeline nothing recomputes anything, so this is the only check that can
show a distributed answer is *right* rather than merely self-consistent. It
needs a model one device can hold, which is why it is optional.

Without a browser:

```bash
npm test
```

## When it goes wrong

- **"the ring stalled at token N"** — a stage went quiet. Press Generate again
  to re-plan around whoever is still connected.
- **"unsupported architecture"** — the model is not Llama-style or
  GPT-2-style.
- **"tokenizer type … is not supported"** — the repo uses SentencePiece or
  WordPiece; only byte-level BPE is implemented.
- **401 / 403** — the model needs a token, or the token lacks access. For
  gated repos, accept the licence on the model page first.
- **"the server ignored the byte range"** — the whole file came back instead of
  a slice; refused rather than accepted, since that defeats the point.
- **"REFUSED activation … belongs to model X"** — a stage still holds a slice
  of a previous model. Reload that device.
- **"⚠ … disagrees with this device's kernel probe"** — that device's
  arithmetic differs from yours. Do not give it layers.
- **out of memory** — the stage's slice does not fit. Add devices, or pick a
  smaller model; the plan shows the per-device cost before you start.

## More

- [ARCHITECTURE.md](ARCHITECTURE.md) — the ring, the plan, the wire protocol.
- The parent projects:
  [DaisyChain-Train](https://huggingface.co/DaisyChainAI/DaisyChain-Train) ·
  [DaisyChain-Web](https://huggingface.co/spaces/Quazim0t0/DaisyChain-Web)
