// Shared training math — pure JS. Used as the WebGPU fallback in the browser,
// and unit-tested directly in Node. Model: linear regression Y = X @ W (MSE).
// The GEMMs here are exactly what the WebGPU compute shader replaces.
(function (root) {
  "use strict";

  // C(m×n) = A(m×k) @ B(k×n), all Float32Array row-major
  function matmul(A, B, m, k, n) {
    const C = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const a = A[i * k + p];
        if (a === 0) continue;
        const bo = p * n, co = i * n;
        for (let j = 0; j < n; j++) C[co + j] += a * B[bo + j];
      }
    }
    return C;
  }

  function transpose(A, rows, cols) {
    const T = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) T[j * rows + i] = A[i * cols + j];
    return T;
  }

  // Forward + loss + gradient for one shard.
  //   X: n×din,  W: din×dout,  y: n×dout
  //   gradW = (2/n) * Xᵀ @ (X@W - y)   (din×dout)
  // matmulFn lets the browser swap in the WebGPU GEMM (same signature as matmul).
  function forwardLossGrad(X, y, W, n, din, dout, matmulFn) {
    const mm = matmulFn || matmul;
    const pred = mm(X, W, n, din, dout);          // n×dout   (GEMM)
    const resid = new Float32Array(n * dout);
    let loss = 0;
    for (let i = 0; i < n * dout; i++) {
      const r = pred[i] - y[i];
      resid[i] = r; loss += r * r;
    }
    loss /= (n * dout);
    const Xt = transpose(X, n, din);              // din×n
    const g = mm(Xt, resid, din, n, dout);        // din×dout (GEMM)
    const scale = 2 / n;
    for (let i = 0; i < g.length; i++) g[i] *= scale;
    return { pred, loss, gradW: g };
  }

  function applyGrad(W, gradAvg, lr) {
    for (let i = 0; i < W.length; i++) W[i] -= lr * gradAvg[i];
  }

  // average a list of gradient Float32Arrays (equal weight)
  // ORDER MATTERS: float addition is not associative, so every replica MUST
  // pass the gradients in the same order (the leader's roster order) or their
  // averages differ in the last bits and the weights fork. Never self-first.
  //
  // REFUSES (throws GradientError) rather than averaging when (a) the gradients differ in
  // length -- typed arrays silently ignore out-of-range writes, so a truncated gradient used
  // to be averaged over a prefix with no error -- or (b) any value is non-finite: one NaN
  // would enter DaisyAdam's m/v on EVERY peer (and every later checkpoint) identically, so no
  // fork check could see it. Refusing BEFORE the optimizer keeps weights and moments clean.
  function averageGrads(grads) {
    const len = grads[0].length;
    for (let r = 0; r < grads.length; r++) {
      const g = grads[r];
      if (!g || g.length !== len) throw gradError("gradient " + r + " has " +
        (g ? g.length : "no") + " values, gradient 0 has " + len);
      for (let i = 0; i < len; i++) if (!Number.isFinite(g[i]))
        throw gradError("gradient " + r + " is non-finite at index " + i + " (" + g[i] + ")");
    }
    const out = new Float32Array(len);
    for (const g of grads) for (let i = 0; i < g.length; i++) out[i] += g[i];
    for (let i = 0; i < out.length; i++) out[i] /= grads.length;
    return out;
  }

  function gradError(msg) { const e = new Error(msg); e.name = "GradientError"; return e; }

  // DaisyAdam — Adam with bias correction, applied to the cluster-averaged
  // gradient. State is a pure function of the gradient sequence, so every peer
  // that averages the same gradients keeps bit-identical moments: no optimizer
  // state ever crosses the wire. Momentum also smooths the noisy STE gradients
  // coming out of the verified INT8 units.
  function makeAdam(dim, opts) {
    const o = opts || {};
    const lr = o.lr ?? 0.02, b1 = o.beta1 ?? 0.9, b2 = o.beta2 ?? 0.999, eps = o.eps ?? 1e-8;
    const m = new Float32Array(dim), v = new Float32Array(dim);
    let t = 0;
    return {
      name: `adam(lr=${lr})`,
      // returns the update u; caller does W[i] -= u[i]
      step(g) {
        t++;
        const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
        const u = new Float32Array(dim);
        for (let i = 0; i < dim; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i];
          v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          u[i] = lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
        }
        return u;
      },
      // snapshot/restore the moments — this is what lets a device that joins
      // mid-run become bit-identical with the group (weights alone are not
      // enough; Adam's m/v/t must match too)
      getState() { return { m: Float32Array.from(m), v: Float32Array.from(v), t }; },
      setState(s) { m.set(s.m); v.set(s.v); t = s.t; },
    };
  }

  const api = { matmul, transpose, forwardLossGrad, applyGrad, averageGrads, makeAdam };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TrainCore = api;
})(typeof self !== "undefined" ? self : this);
