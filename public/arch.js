// config.json -> a normalized model spec, and the tensor names to fetch.
//
// "Load any model off HuggingFace" is not a file-format problem, it is an
// architecture problem: the weights are easy to read, but a Llama block and a
// GPT-2 block are different computations. This file is where that difference
// is confined. Everything downstream — sharding, the ring, the kernels —
// works off the normalized spec and never asks what family a model came from.
//
// Supported families, and what each one needs from the forward pass:
//
//   llama / mistral / qwen2 / smollm / tinyllama   RMSNorm, RoPE, GQA, SwiGLU
//   gpt2 / gpt2-style                              LayerNorm+bias, learned
//                                                  positions, fused QKV, GELU
//
// Anything else is refused BY NAME with a clear message. Silently treating an
// unknown architecture as a known one would produce fluent, confident, wrong
// output — the exact failure this project spends its verification budget on
// making impossible.
(function (root) {
  "use strict";

  const LLAMA_LIKE = new Set(["llama", "mistral", "qwen2", "qwen3", "smollm", "tinyllama", "olmo", "phi3"]);
  const GPT2_LIKE = new Set(["gpt2", "gpt_neo"]);

  function pick(cfg, keys, dflt) {
    for (const k of keys) if (cfg[k] != null) return cfg[k];
    return dflt;
  }

  function fromConfig(cfg) {
    const type = String(cfg.model_type || "").toLowerCase();
    const family = LLAMA_LIKE.has(type) ? "llama" : GPT2_LIKE.has(type) ? "gpt2" : null;
    if (!family)
      throw new Error(`unsupported architecture "${cfg.model_type || "unknown"}". ` +
        `This build implements Llama-style (RMSNorm/RoPE/GQA/SwiGLU) and GPT-2-style blocks. ` +
        `Running a different block shape through these kernels would produce confident nonsense, so it is refused.`);

    const hidden = pick(cfg, ["hidden_size", "n_embd"], 0);
    const layers = pick(cfg, ["num_hidden_layers", "n_layer"], 0);
    const heads = pick(cfg, ["num_attention_heads", "n_head"], 0);
    const vocab = pick(cfg, ["vocab_size"], 0);
    if (!hidden || !layers || !heads || !vocab)
      throw new Error("config.json is missing hidden_size / num_hidden_layers / num_attention_heads / vocab_size");

    const kvHeads = family === "llama" ? pick(cfg, ["num_key_value_heads"], heads) : heads;
    const headDim = pick(cfg, ["head_dim"], Math.floor(hidden / heads));
    if (heads * headDim !== hidden && family === "gpt2")
      throw new Error(`head geometry does not divide the hidden size (${heads} heads x ${headDim} != ${hidden})`);

    const spec = {
      family, modelType: type, hidden, layers, heads, kvHeads, headDim, vocab,
      inter: pick(cfg, ["intermediate_size", "n_inner"], family === "gpt2" ? 4 * hidden : 4 * hidden),
      maxPos: pick(cfg, ["max_position_embeddings", "n_positions", "n_ctx"], 2048),
      normEps: pick(cfg, ["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"], 1e-5),
      norm: family === "llama" ? "rms" : "ln",
      act: family === "llama" ? "silu" : "gelu",
      gated: family === "llama",                 // SwiGLU has a gate branch; GELU MLP does not
      rope: family === "llama",
      ropeTheta: pick(cfg, ["rope_theta"], 10000),
      tie: pick(cfg, ["tie_word_embeddings"], family === "gpt2"),
      qkvFused: family === "gpt2",               // GPT-2 packs q,k,v in one c_attn
      bias: family === "gpt2",                   // Llama-likes are bias-free
      // GPT-2's Conv1D stores (in, out) — already the k x n our GEMM wants.
      // torch.nn.Linear stores (out, in) and must be transposed at load.
      weightLayout: family === "gpt2" ? "in_out" : "out_in",
    };
    if (spec.gated && !cfg.intermediate_size)
      throw new Error("a gated MLP needs intermediate_size in config.json");
    if (spec.kvHeads && spec.heads % spec.kvHeads !== 0)
      throw new Error(`num_attention_heads (${spec.heads}) is not a multiple of num_key_value_heads (${spec.kvHeads})`);
    return spec;
  }

  // ---- tensor names -----------------------------------------------------------
  // Returned as { role: name }. Roles are what the forward pass asks for, so a
  // new family only has to be named here, never plumbed through.
  function layerTensors(spec, i) {
    if (spec.family === "llama") {
      const p = `model.layers.${i}.`;
      // The q/k/v biases are listed but OPTIONAL, and that is not a detail:
      // Qwen2.5 ships them while SmolLM does not, and neither says so in
      // config.json (`attention_bias` is simply absent from Qwen's). Trusting
      // the config would silently drop three bias vectors per layer — a model
      // that still generates fluent text, just not the right text. So bias is
      // decided by what the weight file actually contains.
      return {
        nrm1: p + "input_layernorm.weight",
        nrm2: p + "post_attention_layernorm.weight",
        Wq: p + "self_attn.q_proj.weight", bq: p + "self_attn.q_proj.bias",
        Wk: p + "self_attn.k_proj.weight", bk: p + "self_attn.k_proj.bias",
        Wv: p + "self_attn.v_proj.weight", bv: p + "self_attn.v_proj.bias",
        Wo: p + "self_attn.o_proj.weight", bo: p + "self_attn.o_proj.bias",
        Wgate: p + "mlp.gate_proj.weight", Wup: p + "mlp.up_proj.weight", Wdown: p + "mlp.down_proj.weight",
      };
    }
    const p = `h.${i}.`;                          // gpt2 (also seen as transformer.h.N)
    return {
      nrm1: p + "ln_1.weight", nrm1b: p + "ln_1.bias",
      nrm2: p + "ln_2.weight", nrm2b: p + "ln_2.bias",
      Wqkv: p + "attn.c_attn.weight", bqkv: p + "attn.c_attn.bias",
      Wo: p + "attn.c_proj.weight", bo: p + "attn.c_proj.bias",
      Wfc: p + "mlp.c_fc.weight", bfc: p + "mlp.c_fc.bias",
      Wdown: p + "mlp.c_proj.weight", bdown: p + "mlp.c_proj.bias",
    };
  }
  function headTensors(spec) {
    if (spec.family === "llama")
      return { emb: "model.embed_tokens.weight", nrmF: "model.norm.weight",
               lmHead: spec.tie ? null : "lm_head.weight" };
    return { emb: "wte.weight", pos: "wpe.weight",
             nrmF: "ln_f.weight", nrmFb: "ln_f.bias",
             lmHead: spec.tie ? null : "lm_head.weight" };
  }

  // Repos differ on whether GPT-2 tensors carry a "transformer." prefix, and
  // some Llama exports drop "model.". Rather than guessing, resolve each name
  // against the tensors the file actually contains.
  function resolver(available) {
    const has = (n) => available.has(n);
    return function resolve(name, optional) {
      if (name == null) return null;
      if (has(name)) return name;
      for (const alt of ["transformer." + name, name.replace(/^model\./, ""), "model." + name,
                         name.replace(/^transformer\./, "")])
        if (has(alt)) return alt;
      if (optional) return null;
      throw new Error(`the weights do not contain "${name}" — this repo's layout is not one this build recognises`);
    };
  }

  // Every tensor one stage needs: its layers, plus the head's pieces if it is
  // the head. This is the list that becomes byte ranges.
  function tensorsFor(spec, available, st) {
    const resolve = resolver(available);
    const want = [];
    const add = (n, opt) => { const r = resolve(n, opt); if (r) want.push(r); };
    if (st.head) {
      const h = headTensors(spec);
      add(h.emb); add(h.pos, true); add(h.nrmF, true); add(h.nrmFb, true); add(h.lmHead, true);
    }
    for (let l = st.lo; l < st.hi; l++) {
      const t = layerTensors(spec, l);
      // bias roles (b*) are optional everywhere: present in the file or not
      // applied at all. Weight roles (W*, nrm*) are required.
      for (const [role, n] of Object.entries(t)) add(n, /^b/.test(role));
    }
    return want;
  }

  const api = { fromConfig, layerTensors, headTensors, tensorsFor, resolver, LLAMA_LIKE, GPT2_LIKE };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Arch = api;
})(typeof self !== "undefined" ? self : this);
