// The two models, each in its own Web Worker, and the scoring and sampling built on them.
// Scoring follows prewright.evaluation.scoring exactly: <|endoftext|> + context, then the
// continuation (each encoded separately), and the sum of the continuation's token log-probs.

export const MAX_LEN = 512;

export class ModelWorker {
  constructor(name, url, { expectedBytes = 0, onProgress = () => {}, onStatus = () => {} } = {}) {
    this.name = name;
    this.url = new URL(url, globalThis.location?.href).href; // the worker resolves URLs against js/
    this.pending = new Map();
    this.nextId = 1;
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") onProgress(m.loaded, m.total);
      else if (m.type === "status") onStatus(m.text);
      else if (m.type === "reply") {
        const p = this.pending.get(m.id);
        if (!p) return;
        this.pending.delete(m.id);
        m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
      }
    };
    this.failed = new Promise((_, reject) => {
      this.worker.onerror = (e) => {
        e.preventDefault?.();
        const err = new Error(e.message || "the model worker failed to start (module workers or the "
                              + "onnxruntime-web download may be blocked)");
        for (const p of this.pending.values()) p.reject(err);
        this.pending.clear();
        reject(err);
      };
    });
    this.failed.catch(() => {});
    this.ready = this.request({ type: "load", url: this.url, expectedBytes, maxLen: MAX_LEN });
  }

  request(msg) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id });
    });
  }

  score(ids, n) { return this.request({ type: "score", ids, n }); }

  async next(ids) { return (await this.request({ type: "next", ids })).row; }
}

// prewright.evaluation.scoring.encode_pair
export function encodePair(tok, context, continuation, maxLen = MAX_LEN) {
  let ctx = [tok.eotId, ...tok.encode(context)];
  const cont = tok.encode(continuation);
  if (!cont.length) throw new Error("an ending is empty");
  const overflow = ctx.length + cont.length - maxLen;
  if (overflow > 0) ctx = ctx.slice(overflow);
  return { ids: ctx.concat(cont), n: cont.length, cont };
}

// Both endings under both models. gap = log p(A) - log p(B); factor = exp(gap_control - gap_main),
// i.e. how many times more surprised the flight-free model is by A (relative to B) than the control.
export async function comparePair(tok, models, context, endingA, endingB) {
  const a = encodePair(tok, context, endingA);
  const b = encodePair(tok, context, endingB);
  const out = { a: { tokens: tok.pieces(a.cont) }, b: { tokens: tok.pieces(b.cont) } };
  const runs = await Promise.all(models.map(async (m) => {
    const [sa, sb] = await Promise.all([m.score(a.ids, a.n), m.score(b.ids, b.n)]);
    return { name: m.name, lpA: sa.logprob, lpB: sb.logprob, tokA: sa.tokens, tokB: sb.tokens,
             gap: sa.logprob - sb.logprob };
  }));
  for (const r of runs) out[r.name] = r;
  out.diff = out.main.gap - out.control.gap; // same sign convention as eval.json "diff"
  out.factor = Math.exp(-out.diff);
  return out;
}

// --- sampling -------------------------------------------------------------------------------

// Small, fast, seedable PRNG (mulberry32): the same seed gives the same draws.
export function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// One token from next-token log-probs, with temperature and top-p (nucleus) as in
// prewright.model.GPT.generate: temperature 0 is greedy; top-p keeps the smallest set of most
// likely tokens whose probability reaches top_p (the token that crosses it is kept).
export function sample(logprobs, { temperature = 1, topP = 1 }, random) {
  const V = logprobs.length;
  if (!(temperature > 0)) {
    let best = 0;
    for (let i = 1; i < V; i++) if (logprobs[i] > logprobs[best]) best = i;
    return best;
  }
  let mx = -Infinity;
  for (let i = 0; i < V; i++) mx = Math.max(mx, logprobs[i]);
  const p = new Float64Array(V);
  let z = 0;
  for (let i = 0; i < V; i++) { p[i] = Math.exp((logprobs[i] - mx) / temperature); z += p[i]; }
  const order = Array.from({ length: V }, (_, i) => i).sort((i, j) => p[j] - p[i] || i - j);
  let kept = V, cum = 0;
  if (topP < 1) {
    for (let k = 0; k < V; k++) {
      const q = p[order[k]] / z;
      if (cum > topP) { kept = k; break; } // cum - q > top_p for this token: drop it and the rest
      cum += q;
    }
  }
  let mass = 0;
  for (let k = 0; k < kept; k++) mass += p[order[k]];
  let u = random() * mass;
  for (let k = 0; k < kept; k++) {
    u -= p[order[k]];
    if (u <= 0) return order[k];
  }
  return order[kept - 1];
}
