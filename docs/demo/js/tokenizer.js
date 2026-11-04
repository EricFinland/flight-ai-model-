// Byte-level BPE, reading a Hugging Face `tokenizers` tokenizer.json (GPT-2 style:
// ByteLevel pre-tokenizer without prefix space, BPE model, ByteLevel decoder).
// Written to give exactly the ids of prewright.tokenizer.Tokenizer.encode
// (tests/test_demo_tokenizer.py checks this). No dependencies; runs in browsers and Node.

// GPT-2's reversible byte <-> printable-character map.
function byteMaps() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const enc = new Array(256);
  const dec = new Map();
  bs.forEach((b, i) => { enc[b] = String.fromCodePoint(cs[i]); dec.set(String.fromCodePoint(cs[i]), b); });
  return { enc, dec };
}

// The ByteLevel pre-tokenizer's split pattern (GPT-2's), with Unicode classes.
const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

export class Tokenizer {
  constructor(json) {
    const model = json.model;
    if (!model || model.type !== "BPE") throw new Error("tokenizer.json: expected a BPE model");
    this.vocab = new Map(Object.entries(model.vocab));
    this.idToToken = [];
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;
    this.ranks = new Map();
    model.merges.forEach((m, i) => {
      const pair = Array.isArray(m) ? m : m.split(" ");
      this.ranks.set(pair[0] + " " + pair[1], i);
    });
    this.added = new Map();
    for (const t of json.added_tokens || []) {
      this.added.set(t.content, t.id);
      this.idToToken[t.id] = t.content;
    }
    this.special = new Set((json.added_tokens || []).filter(t => t.special).map(t => t.id));
    const names = [...this.added.keys()].sort((a, b) => b.length - a.length).map(escapeRe);
    this.addedRe = names.length ? new RegExp("(" + names.join("|") + ")") : null;
    const { enc, dec } = byteMaps();
    this.byteEnc = enc;
    this.byteDec = dec;
    this.utf8 = new TextEncoder();
    this.cache = new Map();
    this.eotId = this.added.get("<|endoftext|>");
  }

  static async load(url, fetchFn = globalThis.fetch) {
    const r = await fetchFn(url);
    if (!r.ok) throw new Error(`could not load ${url}: HTTP ${r.status}`);
    return new Tokenizer(await r.json());
  }

  get vocabSize() { return this.idToToken.length; }

  // BPE on one pre-token (already mapped to byte characters); returns token strings.
  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1];
      const merged = [];
      for (let i = 0; i < parts.length;) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i += 2; }
        else { merged.push(parts[i]); i += 1; }
      }
      parts = merged;
    }
    if (this.cache.size > 20000) this.cache.clear();
    this.cache.set(word, parts);
    return parts;
  }

  encodeOrdinary(text, out) {
    for (const m of text.matchAll(SPLIT)) {
      let mapped = "";
      for (const b of this.utf8.encode(m[0])) mapped += this.byteEnc[b];
      for (const piece of this.bpe(mapped)) {
        const id = this.vocab.get(piece);
        if (id === undefined) throw new Error(`token not in vocabulary: ${JSON.stringify(piece)}`);
        out.push(id);
      }
    }
  }

  // Text -> ids. Special-token strings (e.g. "<|endoftext|>") become their ids, as in Python.
  encode(text) {
    const out = [];
    const segments = this.addedRe ? text.split(this.addedRe) : [text];
    segments.forEach((seg, i) => {
      if (!seg) return;
      if (i % 2 === 1) out.push(this.added.get(seg));
      else this.encodeOrdinary(seg, out);
    });
    return out;
  }

  // The token strings for ids, decoded one at a time ("Ġballoon" -> " balloon").
  pieces(ids) { return ids.map(id => this.decode([id])); }

  decode(ids, { skipSpecial = false } = {}) {
    const bytes = [];
    for (const id of ids) {
      const tok = this.idToToken[id];
      if (tok === undefined) continue;
      if (this.added.has(tok)) {  // added tokens are plain text, not byte-mapped
        if (skipSpecial && this.special.has(id)) continue;
        for (const b of this.utf8.encode(tok)) bytes.push(b);
        continue;
      }
      for (const ch of tok) {
        const b = this.byteDec.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
  }
}
