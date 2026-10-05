"""Held-out perplexity: flight passages neither model trained on, against ordinary held-out text.

Three piece sets are scored with both models, every piece starting fresh after
``<|endoftext|>``:

``flight``
    Passages cut by the flight scrub from *validation* books
    (``corpus/flight_eval.jsonl``), chunked to the context length.
``ordinary_matched``
    Ordinary held-out text (``corpus/main/val.jsonl``), one piece per flight
    piece with exactly the same token length, each starting at a paragraph
    boundary. Short pieces have higher loss than long ones (less context), so
    matching lengths keeps the comparison fair.
``ordinary_full``
    Full-context windows of ordinary held-out text: the usual validation loss,
    to show the two models are equally fluent in general.

The headline statistic is the *excess* loss of main over control on flight
text, minus the same excess on matched ordinary text. Inside flight pieces the
loss is also split between the tokens of flight-lexicon words themselves
("balloon", "aeronaut"...) and all other tokens.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from prewright.evaluation import stats
from prewright.evaluation.interface import Codec, LanguageModel, with_bos

_PARA = re.compile(r"\n[ \t]*\n")
_FALLBACK_FLIGHT = re.compile(
    r"balloon\w*|aeronaut\w*|aerostat\w*|air-?ships?|parachut\w*|flying[- ]machines?|aeroplanes?|"
    r"dirigibles?|zeppelin\w*|montgolfier\w*|gliders?|aviat\w*", re.IGNORECASE)


def flight_pattern() -> re.Pattern[str]:
    """The project's unconditional flight-lexicon pattern, or a small fallback."""
    try:
        from prewright.flight import FLIGHT_PATTERN
        return FLIGHT_PATTERN
    except ImportError:  # pragma: no cover - flight.py exists in the repo
        return _FALLBACK_FLIGHT


@dataclass
class Piece:
    ids: list[int]
    text: str
    source: str = ""
    flight_mask: list[bool] = field(default_factory=list)  # per token: inside a flight-lexicon word


def iter_texts(path: str | Path, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """(id, text) from a JSONL file. Accepts "text", or a list in "paragraphs"/"passages"."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("text")
            if text is None:
                parts = r.get("paragraphs") or r.get("passages") or []
                text = "\n\n".join(p if isinstance(p, str) else p.get("text", "") for p in parts)
            if not text or not text.strip():
                continue
            yield str(r.get("id", r.get("book_id", k))), text
            n += 1
            if limit is not None and n >= limit:
                return


def _token_char_spans(codec: Codec, text: str, ids: Sequence[int]) -> list[tuple[int, int]]:
    """Character span of each token in ``text``.

    Uses the tokenizer's offsets when it has them; otherwise the lengths of
    tokens decoded one by one (exact for ASCII text, which PG-19 almost is).
    """
    enc_offsets = getattr(codec, "encode_offsets", None)
    if enc_offsets is not None:
        got_ids, offsets = enc_offsets(text)
        if list(got_ids) == list(ids):
            return list(offsets)
    spans, pos = [], 0
    for t in ids:
        n = len(codec.decode([t]))
        spans.append((pos, pos + n))
        pos += n
    return spans


def flight_token_mask(codec: Codec, text: str, ids: Sequence[int], pattern: re.Pattern[str] | None = None) -> list[bool]:
    pattern = pattern or flight_pattern()
    hits = [m.span() for m in pattern.finditer(text)]
    if not hits:
        return [False] * len(ids)
    return [any(s < he and e > hs for hs, he in hits) for s, e in _token_char_spans(codec, text, ids)]


def make_flight_pieces(codec: Codec, texts: Sequence[tuple[str, str]], max_tokens: int, min_tokens: int,
                       max_pieces: int, seed: int, max_per_source: int | None = None) -> list[Piece]:
    """Chunk flight passages into pieces of at most ``max_tokens``; keep a seeded random sample.

    ``max_per_source`` caps pieces per source book, so one validation book that is
    *about* ballooning cannot dominate the set (pieces from one book are not independent).
    """
    pattern = flight_pattern()
    pieces = []
    for src, text in texts:
        ids = codec.encode(text)
        for start in range(0, len(ids), max_tokens):
            window = ids[start:start + max_tokens]
            if len(window) < min_tokens:
                continue
            piece_text = codec.decode(window)
            pieces.append(Piece(window, piece_text, src, flight_token_mask(codec, piece_text, window, pattern)))
    random.Random(seed).shuffle(pieces)
    if max_per_source is not None:
        per: dict[str, int] = {}
        kept = []
        for p in pieces:
            per[p.source] = per.get(p.source, 0) + 1
            if per[p.source] <= max_per_source:
                kept.append(p)
        pieces = kept
    return pieces[:max_pieces]


def make_matched_pieces(codec: Codec, books: Sequence[tuple[str, str]], lengths: Sequence[int], seed: int,
                        exclude: re.Pattern[str] | None = None, tries: int = 50) -> list[Piece]:
    """One ordinary piece of exactly ``L`` tokens per requested length, starting at a paragraph boundary."""
    rng = random.Random(seed)
    paras = [(src, [p for p in _PARA.split(t) if p.strip()]) for src, t in books]
    paras = [(s, ps) for s, ps in paras if ps]
    if not paras:
        return []
    out = []
    for length in lengths:
        for _ in range(tries):
            src, ps = rng.choice(paras)
            i = rng.randrange(len(ps))
            buf, chars = [], 0
            while i < len(ps) and chars < length * 8:
                buf.append(ps[i])
                chars += len(ps[i])
                i += 1
            ids = codec.encode("\n\n".join(buf))
            if len(ids) < length:
                continue
            window = ids[:length]
            text = codec.decode(window)
            if exclude is not None and exclude.search(text):
                continue
            out.append(Piece(window, text, src, [False] * length))
            break
    return out


def score_pieces(lm: LanguageModel, codec: Codec, pieces: Sequence[Piece]) -> list[dict]:
    lps = lm.token_logprobs([with_bos(codec, p.ids) for p in pieces])
    rows = []
    for p, lp in zip(pieces, lps):
        lp = lp[-len(p.ids):]  # scores of the piece's own tokens (the first one is scored after <eot>)
        mask = p.flight_mask or [False] * len(p.ids)
        rows.append({
            "nll": -float(sum(lp)), "n_tokens": len(p.ids), "n_bytes": len(p.text.encode("utf-8")),
            "nll_flight_tokens": -float(sum(x for x, m in zip(lp, mask) if m)),
            "n_flight_tokens": int(sum(mask)),
        })
    return rows


def _set_summary(rows: list[dict], seed: int) -> dict:
    nll = [r["nll"] for r in rows]
    tok = [r["n_tokens"] for r in rows]
    total_nll, total_tok, total_bytes = sum(nll), sum(tok), sum(r["n_bytes"] for r in rows)
    return {
        "pieces": len(rows), "tokens": total_tok, "bytes": total_bytes,
        "loss": total_nll / total_tok if total_tok else float("nan"),
        "perplexity": stats.bootstrap_ratio(nll, tok, seed=seed, transform=np.exp).to_dict() if rows else None,
        "bits_per_byte": total_nll / math.log(2) / total_bytes if total_bytes else float("nan"),
    }


def _excess(main: list[dict], ctrl: list[dict]) -> list[float]:
    """Per piece: mean per-token NLL of main minus control."""
    return [(m["nll"] - c["nll"]) / m["n_tokens"] for m, c in zip(main, ctrl)]


def analyze_perplexity(sets: dict[str, dict[str, list[dict]]], main: str = "main", control: str = "control",
                       seed: int = 0, flight_sources: Sequence[str] | None = None) -> dict:
    """``sets[set_name][model_name]`` is a list of score_pieces rows (same pieces, same order).

    ``flight_sources`` (source book of each flight piece) adds a book-clustered interval.
    """
    out: dict = {"sets": {}}
    for name, by_model in sets.items():
        m, c = by_model[main], by_model[control]
        excess = stats.bootstrap_ratio([a["nll"] - b["nll"] for a, b in zip(m, c)], [a["n_tokens"] for a in m],
                                       seed=seed)
        out["sets"][name] = {
            main: _set_summary(m, seed), control: _set_summary(c, seed),
            # token-weighted mean NLL difference (nats/token), and as a perplexity ratio
            "excess_loss": excess.to_dict(),
            "perplexity_ratio": {"estimate": math.exp(excess.estimate), "low": math.exp(excess.low),
                                 "high": math.exp(excess.high)} if m else None,
            "frac_pieces_main_worse": float(np.mean([x > 0 for x in _excess(m, c)])) if m else float("nan"),
        }
    if "flight" in sets and "ordinary_matched" in sets:
        f = _excess(sets["flight"][main], sets["flight"][control])
        o = _excess(sets["ordinary_matched"][main], sets["ordinary_matched"][control])
        out["flight_minus_ordinary_excess"] = {
            "diff": stats.bootstrap_diff_of_means(f, o, seed=seed).to_dict(),
            "permutation_p": stats.two_sample_permutation_test(f, o, seed=seed),
            "diff_book_clustered": stats.bootstrap_diff_of_means(f, o, seed=seed, clusters_x=flight_sources).to_dict()
            if flight_sources is not None else None,
            "note": "per-piece mean excess NLL (main - control, nats/token) on flight pieces minus on "
                    "length-matched ordinary pieces",
        }
        fm, fc = sets["flight"][main], sets["flight"][control]
        n_ft = sum(r["n_flight_tokens"] for r in fm)
        if n_ft:
            other_m = sum(r["nll"] - r["nll_flight_tokens"] for r in fm)
            other_c = sum(r["nll"] - r["nll_flight_tokens"] for r in fc)
            n_other = sum(r["n_tokens"] - r["n_flight_tokens"] for r in fm)
            have = [k for k, r in enumerate(fm) if r["n_flight_tokens"]]
            ft_ci = stats.bootstrap_ratio([fm[k]["nll_flight_tokens"] - fc[k]["nll_flight_tokens"] for k in have],
                                          [fm[k]["n_flight_tokens"] for k in have], seed=seed)
            out["flight_word_tokens"] = {
                "n_tokens": n_ft, "pieces_with_hits": len(have),
                f"surprisal_bits_{main}": sum(r["nll_flight_tokens"] for r in fm) / n_ft / math.log(2),
                f"surprisal_bits_{control}": sum(r["nll_flight_tokens"] for r in fc) / n_ft / math.log(2),
                "excess_nats_per_token": ft_ci.to_dict(),
                f"other_tokens_surprisal_bits_{main}": other_m / n_other / math.log(2) if n_other else float("nan"),
                f"other_tokens_surprisal_bits_{control}": other_c / n_other / math.log(2) if n_other else float("nan"),
            }
    return out
