"""Write evaluation results: ``<name>.json`` (everything) and ``REPORT.md`` (tables and headline facts).

The markdown is generated from the numbers only. A headline sentence is
written as a finding only when its 95% interval excludes "no difference";
otherwise it is reported as not significant. Every figure carries its sample
size.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAIN, CONTROL = "main", "control"


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(iv: dict | None, nd: int = 2, pct: bool = False) -> str:
    if not iv:
        return "–"
    k = 100 if pct else 1
    unit = "%" if pct else ""
    return f"{_f(iv['estimate'] * k, nd if not pct else 1)}{unit} [{_f(iv['low'] * k, nd if not pct else 1)}, " \
           f"{_f(iv['high'] * k, nd if not pct else 1)}]"


def _p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def _times(x: float) -> str:
    if x is None or math.isnan(x):
        return "–"
    if x >= 1e4:
        e = int(math.floor(math.log10(x)))
        return f"{x / 10 ** e:.1f}×10^{e}"
    return f"{x:.0f}×" if x >= 100 else f"{x:.1f}×" if x >= 10 else f"{x:.2f}×"


def _slot(context: str, target: str) -> str:
    """'context **target**' with the target's leading space outside the bold markers."""
    sep = " " if target[:1].isspace() else ""
    return f"{context}{sep}**{target.strip()}**"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |" for r in rows]
    return out + [""]


def _sig(iv: dict | None) -> bool:
    return bool(iv) and (iv["low"] > 0 or iv["high"] < 0)


# ---------------------------------------------------------------------------
# Headline facts


def headline_facts(res: dict) -> list[str]:
    facts: list[str] = []
    paired = res.get("paired")
    if paired and "flight" in paired.get("kinds", {}):
        d = paired["kinds"]["flight"]["difference"]
        lr = d["likelihood_ratio"]
        n = d["n"]
        if _sig(d["mean_diff"]) and d["mean_diff"]["estimate"] < 0:
            facts.append(
                f"Relative to an everyday 1900 wording, the flight-free model finds the flight wording "
                f"**{_times(lr['geometric_mean'])} less likely** than the control model does (geometric mean over "
                f"{n} matched pairs; 95% CI {_times(lr['low'])}–{_times(lr['high'])}; median {_times(lr['median'])}; "
                f"it is the more surprised of the "
                f"two on {d['frac_main_lower'] * 100:.0f}% of pairs; sign test p {_p(d['sign_test']['p_value'])}).")
        else:
            facts.append(f"On {n} flight pairs the main-vs-control difference is **not significant** "
                         f"(mean {_f(d['mean_diff']['estimate'])} nats, 95% CI {_f(d['mean_diff']['low'])} to "
                         f"{_f(d['mean_diff']['high'])}).")
        if "control" in paired["kinds"]:
            c = paired["kinds"]["control"]["difference"]
            agree = not _sig(c["mean_diff"])
            facts.append(
                f"On {c['n']} non-flight control pairs the two models "
                + ("**agree**" if agree else "**differ**")
                + f": factor {_times(c['likelihood_ratio']['geometric_mean'])} (95% CI "
                f"{_times(c['likelihood_ratio']['low'])}–{_times(c['likelihood_ratio']['high'])})"
                + ("." if agree else "; part of the flight effect may be a general difference between the models."))
        for item in [i for i in paired["items"] if i["headline"]][:4]:
            r = math.exp(-item["diff"])
            facts.append(
                f"\"{_slot(item['context'], item['target'])}\" vs \"…**{item['neutral'].strip()}**\": the flight-free model "
                + (f"finds the flight version {_times(r)} more surprising (relative to the everyday one) than the control does."
                   if r >= 1 else f"is *less* surprised by the flight version than the control ({_times(1 / r)}); single item, noisy."))
    ppl = res.get("perplexity")
    if ppl and "flight" in ppl.get("sets", {}):
        fs, os_ = ppl["sets"]["flight"], ppl["sets"].get("ordinary_matched")
        line = (f"On {fs[MAIN]['pieces']} held-out flight passages (never trained on by either model), perplexity is "
                f"**{_f(fs[MAIN]['perplexity']['estimate'], 1)}** for the flight-free model against "
                f"**{_f(fs[CONTROL]['perplexity']['estimate'], 1)}** for the control "
                f"(ratio {_f(fs['perplexity_ratio']['estimate'])}, 95% CI {_f(fs['perplexity_ratio']['low'])}–"
                f"{_f(fs['perplexity_ratio']['high'])})")
        if os_:
            line += (f"; on length-matched ordinary held-out text the ratio is {_f(os_['perplexity_ratio']['estimate'])} "
                     f"(95% CI {_f(os_['perplexity_ratio']['low'])}–{_f(os_['perplexity_ratio']['high'])}).")
        facts.append(line)
        fw = ppl.get("flight_word_tokens")
        if fw:
            facts.append(f"Inside those passages, the tokens of flight words themselves cost "
                         f"**{_f(fw['surprisal_bits_main'], 1)} bits** each for the flight-free model against "
                         f"{_f(fw['surprisal_bits_control'], 1)} for the control ({fw['n_tokens']} tokens); all other "
                         f"tokens cost {_f(fw['other_tokens_surprisal_bits_main'], 2)} vs "
                         f"{_f(fw['other_tokens_surprisal_bits_control'], 2)} bits.")
    tok = res.get("tokenizer")
    if tok:
        b = next((r for r in tok["rows"] if r["flight"] == "balloon"), None)
        if b:
            fi = b["flight_info"][repr(" balloon")]
            ev = b["everyday"][0]
            facts.append(f"The tokenizer (trained on the scrubbed corpus) splits \" balloon\" into {fi['n_tokens']} "
                         f"pieces {fi['pieces']}, but \" {ev['word']}\" into {ev[repr(' ' + ev['word'])]['n_tokens']} "
                         f"{ev[repr(' ' + ev['word'])]['pieces']}. Over {tok['n_flight']} flight words the mean is "
                         f"{_f(tok['mean_tokens_flight'])} tokens, against {_f(tok['mean_tokens_everyday'])} for "
                         f"{tok['n_everyday']} everyday comparison words.")
    corpus = res.get("corpus") or {}
    if MAIN in corpus and CONTROL in corpus:
        cm, cc = corpus[MAIN]["counts"], corpus[CONTROL]["counts"]
        words = [w for w in ("balloon", "aeronaut", "airship", "parachute", "flying machine", "montgolfier")
                 if cc.get(w)]
        if words:
            facts.append("Exact training-text counts, flight-free vs control: "
                         + "; ".join(f"\"{w}\" {cm.get(w, 0):,} vs {cc.get(w, 0):,}" for w in words) + ".")
        a = corpus[MAIN].get("audit")
        if a is not None:
            facts.append(f"Flight-lexicon audit of the flight-free training text ({corpus[MAIN]['books']:,} books): "
                         f"**{a['flight_hits']} hits** in {a['books_with_hits']} books"
                         + (f" (control: {corpus[CONTROL]['audit']['flight_hits']:,} hits in "
                            f"{corpus[CONTROL]['audit']['books_with_hits']:,} books)." if corpus[CONTROL].get("audit") else "."))
    know = res.get("knowledge")
    if know and "general" in know.get("kinds", {}):
        g = know["kinds"]["general"]
        facts.append(f"It is not simply ignorant: on {g[MAIN]['n']} general 19th-century knowledge cloze items the "
                     f"flight-free model scores {g[MAIN]['correct']}/{g[MAIN]['n']} and the control "
                     f"{g[CONTROL]['correct']}/{g[CONTROL]['n']} (chance ≈ {g[MAIN]['chance'] * 100:.0f}%; "
                     f"McNemar p {_p(g['discordant']['mcnemar_p'])}).")
        if "flight" in know["kinds"]:
            fk = know["kinds"]["flight"]
            facts.append(f"On {fk[MAIN]['n']} flight-history cloze items: flight-free {fk[MAIN]['correct']}/"
                         f"{fk[MAIN]['n']}, control {fk[CONTROL]['correct']}/{fk[CONTROL]['n']} "
                         f"(McNemar p {_p(fk['discordant']['mcnemar_p'])}).")
    gen = res.get("generation")
    if gen and gen.get("summary"):
        cats = gen["summary"]["categories"]
        fe = cats.get("flight_eliciting")
        if fe:
            facts.append(f"Prompted towards flight, the control's completions use a flight-lexicon term in "
                         f"{fe[CONTROL]['with_flight_terms']}/{fe[CONTROL]['n']} samples, the flight-free model's in "
                         f"{fe[MAIN]['with_flight_terms']}/{fe[MAIN]['n']} (prompt words not counted).")
    return facts


# ---------------------------------------------------------------------------
# Sections


def _paired_section(p: dict) -> list[str]:
    L = ["## 1. Paired surprisal probes", "",
         "Each item has one context and two continuations in the same slot. *Gap* = log p(flight wording) − "
         "log p(everyday wording), in nats, per model. *Diff* = gap(main) − gap(control); negative means the "
         "flight-free model is relatively more surprised by the flight wording. The factor is exp(−mean diff): "
         "how many times less likely, relative to the everyday wording, the flight-free model finds the flight "
         "wording than the control does. 95% bootstrap CIs (10,000 resamples of items); two-sided sign test and "
         "sign-flip permutation test on the per-item diffs.", ""]
    rows = []
    for kind, k in p["kinds"].items():
        d = k["difference"]
        rows.append([kind, d["n"], _ci(k[MAIN]["mean_gap"]), _ci(k[CONTROL]["mean_gap"]),
                     _ci(k[MAIN]["prefer_neutral"], pct=True), _ci(k[CONTROL]["prefer_neutral"], pct=True),
                     _ci(d["mean_diff"]), f"{_times(d['likelihood_ratio']['geometric_mean'])} "
                     f"[{_times(d['likelihood_ratio']['low'])}, {_times(d['likelihood_ratio']['high'])}]",
                     _p(d["sign_test"]["p_value"]), _p(d["permutation_p"])])
    L += _table(["pairs", "n", "mean gap main", "mean gap control", "main prefers everyday", "control prefers everyday",
                 "mean diff (nats)", "factor", "sign p", "perm. p"], rows)
    if "flight_vs_control_pairs" in p:
        fc = p["flight_vs_control_pairs"]
        L += [f"Flight-pair diff minus control-pair diff: {_ci(fc['diff_of_mean_diffs'])} nats "
              f"(two-sample permutation p {_p(fc['permutation_p'])}).", ""]
    if "flight" in p["kinds"]:
        d = p["kinds"]["flight"]["difference"]
        L += [f"Where the flight-pair difference comes from: log p(flight wording) moves by "
              f"{_ci(d['target_shift'])} nats (main − control); log p(everyday wording) moves by "
              f"{_ci(d['neutral_shift'])}. Per-token-normalised diff: {_ci(d['mean_diff_per_token'], 3)} nats/token.", ""]
    L += ["### By category", ""]
    rows = [[c["kind"], c["category"], c["n"], _f(c["mean_gap_main"]), _f(c["mean_gap_control"]), _ci(c["mean_diff"]),
             f"{c['frac_main_lower'] * 100:.0f}%"] for c in p["categories"].values()]
    L += _table(["kind", "category", "n", "gap main", "gap control", "diff [95% CI]", "main more surprised"], rows)
    L += ["### Largest effects (flight pairs)", ""]
    rows = [[i["id"], f"{_slot(i['context'], i['target'])} / {i['neutral'].strip()}", _f(i["gap_main"]),
             _f(i["gap_control"]), _times(math.exp(-i["diff"]))] for i in p["top_effects"]]
    L += _table(["id", "context **flight** / everyday", "gap main", "gap control", "factor"], rows)
    L += ["### Smallest effects (flight pairs)", "",
          "Listed for honesty: items where the flight-free model is barely, or not at all, more surprised.", ""]
    rows = [[i["id"], f"{_slot(i['context'], i['target'])} / {i['neutral'].strip()}", _f(i["gap_main"]),
             _f(i["gap_control"]), _times(math.exp(-i["diff"]))] for i in p.get("weakest_effects", [])]
    L += _table(["id", "context **flight** / everyday", "gap main", "gap control", "factor"], rows)
    return L


def _ppl_section(pp: dict) -> list[str]:
    L = ["## 2. Held-out perplexity", "",
         "`flight`: passages the scrub removed from validation books (neither model trained on them). "
         "`ordinary_matched`: ordinary validation text, one piece of identical token length per flight piece. "
         "`ordinary_full`: full-context windows of ordinary validation text. CIs resample pieces.", ""]
    src = pp.get("sources")
    if src:
        L += [f"Flight pieces come from {src.get('flight_source_books', '–')} validation books (at most "
              f"{src.get('max_pieces_per_book', '–')} pieces per book, since pieces from one book are correlated); "
              f"ordinary pieces from {src.get('val_books_read', '–')} validation books; context "
              f"{src.get('context_tokens', '–')} tokens.", ""]
    rows = []
    for name, s in pp["sets"].items():
        rows.append([name, s[MAIN]["pieces"], s[MAIN]["tokens"], _ci(s[MAIN]["perplexity"], 1),
                     _ci(s[CONTROL]["perplexity"], 1), _f(s[MAIN]["bits_per_byte"], 3), _f(s[CONTROL]["bits_per_byte"], 3),
                     f"{_f(s['perplexity_ratio']['estimate'], 3)} [{_f(s['perplexity_ratio']['low'], 3)}, "
                     f"{_f(s['perplexity_ratio']['high'], 3)}]", f"{s['frac_pieces_main_worse'] * 100:.0f}%"])
    L += _table(["set", "pieces", "tokens", "ppl main", "ppl control", "bpb main", "bpb control",
                 "ppl ratio main/control", "pieces where main worse"], rows)
    if "flight_minus_ordinary_excess" in pp:
        x = pp["flight_minus_ordinary_excess"]
        L += [f"Excess loss of main over control on flight pieces minus on matched ordinary pieces: "
              f"{_ci(x['diff'], 3)} nats/token (permutation p {_p(x['permutation_p'])}).", ""]
    if "flight_word_tokens" in pp:
        fw = pp["flight_word_tokens"]
        L += [f"Flight-lexicon word tokens inside flight pieces ({fw['n_tokens']} tokens in {fw['pieces_with_hits']} "
              f"pieces): {_f(fw['surprisal_bits_main'], 2)} bits/token (main) vs {_f(fw['surprisal_bits_control'], 2)} "
              f"(control); excess {_ci(fw['excess_nats_per_token'])} nats/token. All other tokens in those pieces: "
              f"{_f(fw['other_tokens_surprisal_bits_main'], 3)} vs {_f(fw['other_tokens_surprisal_bits_control'], 3)} bits.", ""]
    return L


def _knowledge_section(k: dict) -> list[str]:
    L = ["## 3. Knowledge probes (cloze)", "",
         "The model is right on an item when the correct answer gets a higher summed log-prob than every distractor. "
         "Wilson 95% intervals; exact one-sided binomial test against chance; exact McNemar test on the items "
         "where exactly one model is right.", ""]
    rows = []
    for kind, s in k["kinds"].items():
        rows.append([kind, s[MAIN]["n"], _ci(s[MAIN]["accuracy"], pct=True), _ci(s[CONTROL]["accuracy"], pct=True),
                     f"{s[MAIN]['chance'] * 100:.0f}%", _p(s[MAIN]["p_above_chance"]), _p(s[CONTROL]["p_above_chance"]),
                     f"{s['discordant']['main_only']} / {s['discordant']['control_only']}", _p(s["discordant"]["mcnemar_p"]),
                     f"{s['main_per_token']['correct']} / {s['control_per_token']['correct']}"])
    L += _table(["items", "n", "acc main", "acc control", "chance", "p>chance main", "p>chance control",
                 "only main / only control right", "McNemar p", "per-token-normalised correct (main / control)"], rows)
    rows = [[c["kind"], c["category"], c["n"], c[MAIN], c[CONTROL]] for c in k["categories"].values()]
    L += _table(["kind", "category", "n", "main correct", "control correct"], rows)
    return L


def _generation_section(g: dict) -> list[str]:
    s = g["summary"]
    L = ["## 4. Generation probes", "",
         f"Full side-by-side completions: [`{g.get('markdown', 'generations.md')}`]({g.get('markdown', 'generations.md')}). "
         f"Settings: {', '.join(f'{k}={v}' for k, v in g['settings'].items())}. Counts are completions containing "
         "any flight-lexicon term (`prewright.flight`), prompt excluded.", ""]
    rows = [[cat, f"{v[MAIN]['with_flight_terms']}/{v[MAIN]['n']}", f"{v[CONTROL]['with_flight_terms']}/{v[CONTROL]['n']}"]
            for cat, v in s["categories"].items()]
    rows.append(["**all**", f"{_ci(s['overall'][MAIN], pct=True)}", f"{_ci(s['overall'][CONTROL], pct=True)}"])
    return L + _table(["prompt category", "main", "control"], rows)


def _tokenizer_section(t: dict, corpus: dict | None) -> list[str]:
    L = ["## 5. Flight words: tokenizer and corpus counts", "",
         "Token pieces are for the word mid-sentence (with its leading space). Counts are exact case-insensitive "
         "whole-word matches in the training text.", ""]
    cm = (corpus or {}).get(MAIN, {}).get("counts", {})
    cc = (corpus or {}).get(CONTROL, {}).get("counts", {})
    have = bool(corpus and MAIN in corpus)
    rows = []
    for r in t["rows"]:
        words = [(r["flight"], r["flight_info"], True)] + [(e["word"], e, False) for e in r["everyday"]]
        for w, info, is_f in words:
            f = info[repr(" " + w)]
            row = [f"**{w}**" if is_f else w, f["n_tokens"], " · ".join(repr(x) for x in f["pieces"])]
            if have:
                row += [f"{cm.get(w, 0):,}", f"{cc.get(w, 0):,}"]
            rows.append(row)
    L += _table(["word", "tokens", "pieces"] + (["count main", "count control"] if have else []), rows)
    L += [f"Mean tokens: flight words {_f(t['mean_tokens_flight'])} ({t['single_token_flight']}/{t['n_flight']} "
          f"single-token); everyday words {_f(t['mean_tokens_everyday'])} ({t['single_token_everyday']}/"
          f"{t['n_everyday']} single-token).", ""]
    if have:
        phrases = sorted(k for k in set(cm) | set(cc) if " " in k)
        if phrases:
            L += _table(["phrase", "count main", "count control"],
                        [[k, f"{cm.get(k, 0):,}", f"{cc.get(k, 0):,}"] for k in phrases])
        for name in (MAIN, CONTROL):
            c = corpus[name]
            note = f" (first {c['max_books']} books only)" if c.get("max_books") else ""
            line = f"- `{name}`: {c['books']:,} books, {c['chars']:,} characters{note}."
            if c.get("audit"):
                a = c["audit"]
                line += (f" Flight-lexicon audit: {a['flight_hits']:,} hits in {a['books_with_hits']:,} books"
                         + (f"; top terms {', '.join(f'{k} ({v})' for k, v in list(a['by_term'].items())[:6])}"
                            if a["by_term"] else "") + ".")
            L.append(line)
        L.append("")
    return L


def _validation_section(v: dict) -> list[str]:
    return ["## Probe-set checks", "",
            f"- Flight pairs whose *flight* wording (with its context) matches the scrub lexicon: "
            f"{v['flight_target_hits']}/{v['n_flight']}. The rest use wordings the lexicon does not target "
            "(\"rose into the air\", \"learns to fly\"), and test whether the model generalises beyond removed words.",
            f"- Flight pairs whose context itself names a scrubbed term (deliberate: what does the model expect "
            f"after a name it has never seen?): {v.get('context_hits', 0)} ({', '.join(v.get('context_hit_ids', []))}).",
            f"- Other flight pairs whose *everyday* wording matches the lexicon (should be 0): {v['neutral_hits']}.",
            f"- Control pairs where either wording matches the lexicon (should be 0): {v['control_hits']}.", ""]


CAVEATS = [
    "Small models (~10–30M parameters) trained for a few CPU-hours are weak. Absolute probabilities are noisy; "
    "only the *paired* main-vs-control comparisons are meant to carry weight, and individual items are anecdotes.",
    "Within one model, \"prefers the everyday wording\" is confounded by word frequency and option length; the "
    "main − control difference cancels those because both models share the tokenizer and the rest of the corpus.",
    "The two models differ by one training run each (same seed and config). Part of any difference is run-to-run "
    "noise; the non-flight control pairs and ordinary-text perplexity estimate how large that is.",
    "The scrub drops a paragraph and its neighbours, so text that merely co-occurs with flight (the 1870 siege "
    "of Paris, carrier pigeons, meteorology, Vauxhall Gardens) is also thinner in the main corpus. This mostly "
    "makes everyday wordings on such topics less likely under main too, which shrinks the measured effect.",
    "The tokenizer was trained on the main corpus, so flight words cost more tokens for *both* models. This "
    "slightly handicaps the control on flight text; it does not create the main-vs-control difference.",
    "The probe sets were written for this evaluation, not sampled from the corpus, so they reflect the authors' "
    "idea of 19th-century prose. Knowledge items use sum log-prob, which favours shorter answers; "
    "per-token-normalised accuracy is reported alongside.",
]


def write_report(res: dict, out_dir: str | Path, name: str = "eval") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    meta = res.get("meta", {})
    L = ["# Evaluation report: flight-free model vs control", ""]
    models = meta.get("models", {})
    if models:
        L += _table(["model", "checkpoint", "context length"],
                    [[k, f"`{v.get('path', '')}`", v.get("context_length", "")] for k, v in models.items()])
    if meta.get("command"):
        L += [f"Command: `{meta['command']}`", ""]
    L += [f"Seed {meta.get('seed', '–')}; {'**quick mode** (reduced sample sizes)' if meta.get('quick') else 'full run'}; "
          f"generated {meta.get('timestamp', '')}. Raw numbers: [`{json_path.name}`]({json_path.name}).", ""]
    facts = headline_facts(res)
    if facts:
        L += ["## Did you know?", ""] + [f"- {f}" for f in facts] + [""]
    if res.get("paired"):
        L += _paired_section(res["paired"])
    if res.get("perplexity"):
        L += _ppl_section(res["perplexity"])
    if res.get("knowledge"):
        L += _knowledge_section(res["knowledge"])
    if res.get("generation"):
        L += _generation_section(res["generation"])
    if res.get("tokenizer"):
        L += _tokenizer_section(res["tokenizer"], res.get("corpus"))
    if res.get("probe_validation"):
        L += _validation_section(res["probe_validation"])
    if res.get("skipped"):
        L += ["## Skipped", ""] + [f"- {k}: {v}" for k, v in res["skipped"].items()] + [""]
    L += ["## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    md_path = out_dir / "REPORT.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    return json_path, md_path
