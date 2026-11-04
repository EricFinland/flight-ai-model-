// One model per worker: downloads the ONNX file (reporting progress), runs it with
// onnxruntime-web (WebAssembly, single thread) and answers two requests:
//   score {ids, n}  -> log-prob of the last n tokens, each given everything before it
//   next  {ids}     -> log-probs of the next token after ids (Float32Array, vocab size)
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
// GitHub Pages is not cross-origin isolated, so there is no SharedArrayBuffer for threads.
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

let session = null;
let maxLen = 512;

async function download(url, expected) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  const total = Number(r.headers.get("content-length")) || expected || 0;
  if (!r.body || !r.body.getReader) {
    const buf = new Uint8Array(await r.arrayBuffer());
    postMessage({ type: "progress", loaded: buf.length, total: buf.length });
    return buf;
  }
  const reader = r.body.getReader();
  const chunks = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (loaded - last > 256 * 1024) { last = loaded; postMessage({ type: "progress", loaded, total }); }
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  postMessage({ type: "progress", loaded, total: Math.max(total, loaded) });
  return buf;
}

async function logprobs(ids) {
  const T = ids.length;
  const input = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, T]);
  const out = await session.run({ input_ids: input });
  return out.logprobs.data; // Float32Array, T * vocab
}

async function handle(msg) {
  switch (msg.type) {
    case "load": {
      if (typeof WebAssembly !== "object") throw new Error("this browser has no WebAssembly");
      const bytes = await download(msg.url, msg.expectedBytes);
      postMessage({ type: "status", text: "starting" });
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"],
                                                           graphOptimizationLevel: "all" });
      maxLen = msg.maxLen || maxLen;
      return { ok: true };
    }
    case "score": {
      const { ids, n } = msg;
      if (ids.length > maxLen) throw new Error(`sequence of ${ids.length} tokens exceeds ${maxLen}`);
      const lp = await logprobs(ids);
      const V = lp.length / ids.length;
      const tokens = [];
      let sum = 0;
      for (let t = ids.length - n; t < ids.length; t++) {
        const v = lp[(t - 1) * V + ids[t]];
        tokens.push(v);
        sum += v;
      }
      return { logprob: sum, tokens };
    }
    case "next": {
      const ids = msg.ids.slice(-maxLen);
      const lp = await logprobs(ids);
      const V = lp.length / ids.length;
      const row = lp.slice((ids.length - 1) * V, ids.length * V);
      return { row, transfer: [row.buffer] };
    }
    default:
      throw new Error(`unknown request ${msg.type}`);
  }
}

// Requests are handled one at a time, in order.
let queue = Promise.resolve();
self.onmessage = (e) => {
  const msg = e.data;
  queue = queue.then(async () => {
    try {
      const res = await handle(msg);
      const transfer = res.transfer || [];
      delete res.transfer;
      postMessage({ type: "reply", id: msg.id, ok: true, result: res }, transfer);
    } catch (err) {
      postMessage({ type: "reply", id: msg.id, ok: false, error: String(err && err.message || err) });
    }
  });
};
