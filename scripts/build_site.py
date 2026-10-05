#!/usr/bin/env python3
"""Build the public showcase page: one self-contained HTML file for GitHub Pages.

    python3 scripts/build_site.py --results evals/results --data /home/user/data --out docs/site/index.html

Every number on the page is read from the result files; nothing is typed in:

    <results>/eval.json            scripts/evaluate.py (schema: src/prewright/evaluation/run.py, report.py)
    <results>/curve.json           scripts/eval_curve.py
    <results>/chat_showcase.md     scripts/chat_showcase.py
    <results>/flight_gap_curve.png, loss_curves.png   scripts/eval_curve.py (embedded as data URIs)
    <data>/corpus/stats.json, corpus/manifest.csv, tokenized/{main,control}/train.meta.json

Numbers are formatted by the same code that fills README.md
(``scripts/fill_readme_results.py``), so the page and the README always agree.
A figure whose data is missing is left out (with its tile, row or section),
never replaced by a placeholder. Run it after the evaluation; with no
eval.json the page still builds, from the corpus statistics, and says the
results are pending.

The script refuses to embed results from a ``--fake`` evaluation unless
``--allow-fake`` is given, and then marks the page as a smoke test. The page
needs no network apart from optional Google Fonts (it falls back to Georgia).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import fill_readme_results as fr  # noqa: E402
from prewright.evaluation.report import headline_facts  # noqa: E402

try:  # the scrub lexicon, to highlight flight words in the models' answers
    from prewright.flight import find_flight_hits
except ImportError:  # pragma: no cover
    find_flight_hits = None

REPO_URL = "https://github.com/EricFinland/flight-ai-model-"
BLOB = REPO_URL + "/blob/main/"
# Plot colours of scripts/eval_curve.py, reused as small identity swatches next to model names.
C_MAIN, C_CONTROL = "#2a78d6", "#eb6834"
LOOKUP_ERRORS = (fr.Missing, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, AttributeError,
                 OSError, json.JSONDecodeError)


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def inline_md(s: str) -> str:
    """Escape, then render the **bold**, *italic* and `code` that the report's sentences use."""
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def compact(n: float) -> str:
    """8521984 -> '8.5M', 13926400 -> '13.9M', 1415 -> '1,415'."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,.0f}"


def _usable(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if not s or s == fr.TBD or s.startswith("–") or s.endswith("–") or "––" in s:
        return None
    if re.search(r"\b(nan|inf|None)\b", s):
        return None
    return s


# ---------------------------------------------------------------------------
# Inputs


def parse_chat_showcase(md: str) -> list[dict]:
    """chat_showcase.md -> [{"question", "kind", "samples": [{"main", "control", "main_terms", "control_terms"}]}]."""
    out: list[dict] = []
    cur: dict | None = None
    marker = re.compile(r"\s*\*\*\[flight words: ([^\]]*)\]\*\*\s*$")
    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            cur = {"question": line[3:].strip(), "kind": "", "samples": []}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.fullmatch(r"\*([A-Za-z_]+)\*", line)
        if m:
            cur["kind"] = m.group(1)
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) < 3 or not cells[0].isdigit():
            continue  # header and separator rows
        sample: dict = {}
        for name, cell in (("main", cells[1]), ("control", cells[2])):
            terms: list[str] = []
            mm = marker.search(cell)
            if mm:
                terms = [t.strip() for t in mm.group(1).split(",") if t.strip()]
                cell = cell[:mm.start()]
            cell = cell.replace("\\|", "|").strip()
            if cell == "*(empty)*":
                cell = ""
            sample[name], sample[f"{name}_terms"] = cell, terms
        cur["samples"].append(sample)
    return [q for q in out if q["samples"]]


def png_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG"):
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def is_fake(ev: dict) -> bool:
    return any(m.get("path") == "(fake)" for m in (ev.get("meta", {}).get("models") or {}).values())


# ---------------------------------------------------------------------------
# The page


class Page:
    def __init__(self, results: Path, data: Path | None, allow_fake: bool = False):
        self.results, self.data = results, data
        self.notes: list[str] = []  # what was left out, for stderr
        self.masked = False  # a PG-19 "<DW..>" placeholder was replaced by "[…]" in shown model text
        self.ev = self._json(results / "eval.json")
        self.curve = self._json(results / "curve.json")
        if self.ev and is_fake(self.ev) and not allow_fake:
            raise SystemExit(f"{results / 'eval.json'} is from a --fake evaluation; refusing to publish it "
                             f"(pass --allow-fake for a test build)")
        self.res = fr.Results(self.ev, self.curve, 220, data)
        chat_path = results / "chat_showcase.md"
        self.chat = parse_chat_showcase(chat_path.read_text(encoding="utf-8")) if chat_path.is_file() else []
        self.gap_png = png_data_uri(results / "flight_gap_curve.png")
        self.loss_png = png_data_uri(results / "loss_curves.png")

    def _json(self, path: Path) -> dict | None:
        if not path.is_file():
            self.notes.append(f"{path} not found")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            self.notes.append(f"{path} unreadable ({e})")
            return None

    # -- lookups -------------------------------------------------------------
    def v(self, key: str) -> str | None:
        """A README result key, formatted, or None if its data is missing."""
        fn = self.res.resolver(key)
        if fn is None:
            return None
        try:
            return _usable(fn())
        except LOOKUP_ERRORS:
            return None

    def get(self, fn: Callable[[], Any]) -> Any:
        try:
            x = fn()
        except LOOKUP_ERRORS:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    def stats(self) -> dict | None:
        return self.get(self.res.stats)

    def tokmeta(self, corpus: str) -> dict | None:
        return self.get(lambda: self.res.tokmeta(corpus))

    def model_meta(self, key: str) -> Any:
        return self.get(lambda: self.ev["meta"]["models"]["main"][key])

    def params(self) -> str | None:
        p = self.model_meta("parameters")
        return compact(p) if isinstance(p, (int, float)) and p > 0 else None

    def train_tokens(self) -> tuple[str | None, str | None]:
        """(tokens seen by each run, final step), from the checkpoint meta or else the curve."""
        cm = self.model_meta("checkpoint_meta") or {}
        tok, step = cm.get("tokens"), cm.get("step")
        if not tok:
            last = self.get(lambda: self.curve["steps"][-1]) or {}
            tok, step = last.get("tokens"), step or last.get("step")
        return (compact(tok) if tok else None), (f"{step:,}" if isinstance(step, int) else None)

    def audit(self, corpus: str) -> tuple[str, str] | None:
        """(hits, what was audited) for one corpus, from the evaluation's corpus pass."""
        m = re.fullmatch(r"([\d,]+) \((.*)\)", self.v(f"audit_{corpus}_hits") or "")
        return (m.group(1), m.group(2)) if m else None

    def mode(self) -> str | None:
        return self.get(self.res.eval_mode) if self.ev else None

    # -- pieces ----------------------------------------------------------------
    def highlight(self, text: str, terms: list[str] | None = None) -> str:
        """HTML for a model's answer, with flight-lexicon matches wrapped in <mark>."""
        if not text:
            return '<span class="empty">(no answer)</span>'
        if fr.has_mask(text):
            self.masked = True
            text = fr.unmask(text)
        found: list[str] = []
        if find_flight_hits is not None:
            try:
                found = sorted({h.text for h in find_flight_hits(text)}, key=len, reverse=True)
            except Exception:  # never let the highlighter break the build
                found = []
        spans: list[tuple[int, int]] = []
        for word in found:
            pat = re.compile(r"\s+".join(re.escape(p) for p in word.split()), re.IGNORECASE)
            for m in pat.finditer(text):
                if all(m.end() <= a or m.start() >= b for a, b in spans):
                    spans.append((m.start(), m.end()))
        out, pos = [], 0
        for a, b in sorted(spans):
            out.append(esc(text[pos:a]))
            out.append(f'<mark title="in the flight lexicon">{esc(text[a:b])}</mark>')
            pos = b
        out.append(esc(text[pos:]))
        return "".join(out)

    def who(self, model: str) -> str:
        color = C_MAIN if model == "main" else C_CONTROL
        label = "Flight-free model" if model == "main" else "Control model"
        return f'<span class="key" style="--swatch:{color}" aria-hidden="true"></span>{label}'

    def is_null(self) -> bool | None:
        """True if the headline difference is not distinguishable from noise, False if beyond it, None if unknown."""
        verdict = self.v("surprisal_verdict")
        return None if not verdict else verdict.startswith("not ")

    def short_verdict(self) -> str | None:
        null = self.is_null()
        if null is None:
            return None
        return "not distinguishable from model-to-model noise" if null else "beyond model-to-model noise"

    def first_significant_step(self) -> tuple[bool, Any]:
        """(curve scored at all, first significant step or None)."""
        steps = self.get(lambda: self.curve["steps"]) or []
        if not steps:
            return False, None
        return True, self.get(lambda: self.curve["meta"]["first_significant_step"])

    def balloons_read(self) -> int | None:
        """Rough count of "balloon" the control met in training (derived): slice count x share of slice read."""
        n = self.get(lambda: self.ev["corpus"]["control"]["counts"]["balloon"])
        cm = self.model_meta("checkpoint_meta") or {}
        read = cm.get("tokens") if isinstance(cm, dict) else None
        slice_tok = (self.tokmeta("control") or {}).get("tokens")
        if not (isinstance(n, int) and isinstance(read, (int, float)) and isinstance(slice_tok, (int, float))
                and slice_tok > 0):
            return None
        return round(n * min(1.0, read / slice_tok))

    def knowledge_at_chance(self) -> bool | None:
        """True if neither model is shown above chance on the general-knowledge cloze (p >= 0.05)."""
        ps = [self.get(lambda m=m: self.ev["knowledge"]["kinds"]["general"][m]["p_above_chance"])
              for m in ("main", "control")]
        if any(not isinstance(p, (int, float)) for p in ps):
            return None
        return all(p >= 0.05 for p in ps)

    # -- sections ----------------------------------------------------------------
    def banner(self) -> str:
        if not self.ev:
            return ('<div class="banner" role="note"><strong>Results pending.</strong> The evaluation has not been '
                    'run yet; the figures below are from the corpus build only.</div>')
        if is_fake(self.ev):
            return ('<div class="banner" role="note"><strong>Test build.</strong> These numbers come from a fake '
                    'model used to test the page. They are not results.</div>')
        if self.ev.get("meta", {}).get("quick"):
            return ('<div class="banner" role="note"><strong>First look.</strong> These numbers come from a quick '
                    'evaluation with reduced sample sizes; the full run supersedes them.</div>')
        return ""

    def tiles(self) -> str:
        tiles: list[tuple] = []  # (value, label, note[, verdict])
        ratio = self.v("surprisal_ratio")
        if ratio:
            ci = self.v("surprisal_ratio_ci")
            ctl = self.v("control_pairs_ratio")
            note = (f"95% CI {ci}. " if ci else "") + (f"Non-flight pairs, for comparison: {ctl}." if ctl else "")
            verdict = self.short_verdict()
            label = ("how many times more the flight-free model is surprised by a flight wording, relative to an "
                     "everyday one, than the control is")
            tiles.append((ratio, label, note.strip(), (verdict[0].upper() + verdict[1:]) if verdict else ""))
        audit = self.audit("main")
        if audit:
            ctl = self.audit("control")
            note = f"Audited: {audit[1]}." + (f" The control's text: {ctl[0]}." if ctl else "")
            tiles.append((audit[0], "flight-lexicon hits in its training text", note))
        books = self.v("slice_main_books")
        if books:
            tok = self.v("slice_main_tokens")
            tiles.append((books, "books in its training text", f"{tok} tokens." if tok else ""))
        dropped = self.v("corpus_flight_books")
        if dropped:
            tiles.append((dropped, "books removed whole because flight runs through them", ""))
        paras = self.v("corpus_paragraphs_removed")
        if paras:
            cut = self.v("corpus_books_with_cuts")
            tiles.append((paras, "flight paragraphs cut from the books that were kept",
                          f"From {cut} books." if cut else ""))
        params = self.params()
        if params:
            ctx = self.model_meta("context_length")
            tiles.append((params, "parameters", f"Context of {ctx} tokens." if isinstance(ctx, int) else ""))
        tok, step = self.train_tokens()
        if tok:
            tiles.append((tok, "tokens of text read in training", f"{step} steps; the control read the same "
                          "positions." if step else ""))
        if not tiles:
            return ""
        cells = "".join(
            f'<div class="tile"><div class="tile-value">{esc(t[0])}</div>'
            + (f'<div class="tile-verdict">{esc(t[3])}</div>' if len(t) > 3 and t[3] else "")
            + f'<div class="tile-label">{esc(t[1])}</div>'
            + (f'<div class="tile-note">{esc(t[2])}</div>' if t[2] else "") + "</div>" for t in tiles)
        return f'<div class="tiles" id="figures">{cells}</div>'

    def hero(self) -> str:
        dated = self.v("corpus_books_dated")
        src = ("books published in 1902 or earlier, drawn from the "
               + (f"{dated} such books" if dated else "books of that age")
               + " in DeepMind&rsquo;s PG-19 collection of Project Gutenberg texts")
        return f"""
<header class="masthead">
  <div class="ears"><span>Prewright</span><span>A research note</span><span>Printed before 1903</span></div>
  <h1>An AI that has never heard of flight</h1>
  <p class="deck">A small language model trained from scratch on old books, with every balloon, airship, glider,
  parachute and flying machine taken out, and a twin trained on the same books with them left in.</p>
</header>
{self.banner()}
<section class="lede" aria-label="What this is">
  <p class="dropcap">Prewright learned its English only from {src}. Before it read them, every passage about people
  leaving the ground was removed: balloon ascents at country fairs, aeronauts and their parachutes,
  airships in boys' adventure stories, Tennyson's &ldquo;airy navies&rdquo;, the brothers Montgolfier.
  As far as the model knows, nobody has ever flown or seriously tried to. Birds still fly, and so do
  witches and Icarus. A <em>control</em> model was trained the same way on the same books with the flight
  text left in, so any difference between the two comes from that one absence.</p>
</section>
{self.pilot_verdict()}
{self.tiles()}"""

    def pilot_verdict(self) -> str:
        """The pilot's verdict in four lines, from the data where it can be (docs/results.md has the long form)."""
        if not self.ev:
            return ""
        items = []
        m, c = self.audit("main"), self.audit("control")
        if m:
            items.append(("The scrub works on the data.",
                          f"The flight-free model&rsquo;s training text has <strong>{esc(m[0])}</strong> "
                          "flight-lexicon hits" + (f", against {esc(c[0])} in the control&rsquo;s" if c else "")
                          + "."))
        null = self.is_null()
        ratio, ci = self.v("surprisal_ratio"), self.v("surprisal_ratio_ci")
        factor = (f" The headline factor, {esc(ratio)}" + (f" (95% CI {esc(ci)})" if ci else "")
                  + f", is {esc(self.short_verdict())}." if ratio and null is not None else "")
        if null:
            items.append(("The behaviour is a null at this scale.",
                          "On the pre-registered tests the two models are statistically indistinguishable." + factor
                          + " Held-out perplexity, knowledge questions and sampled text agree, and both models "
                          "write grammatical-looking word salad."))
            params, (tok, _) = self.params(), self.train_tokens()
            why = []
            if params and tok:
                why.append(f"At {esc(params)} parameters and {esc(tok)} tokens of reading the models are far from "
                           "converged")
            if self.knowledge_at_chance():
                why.append("neither is shown to beat chance on general 19th-century knowledge questions")
            seen = self.balloons_read()
            if seen:
                why.append(f"even the control met the word &ldquo;balloon&rdquo; only about {seen:,} times in "
                           "training (estimated from its count in the training slice and the share of the slice "
                           "read)")
            if why:
                text = "; ".join(why)
                items.append(("Why.", text[0].upper() + text[1:] + ". The control saw too little flight for the "
                              "two models to disagree about it, so the test was underpowered, not negative."))
            items.append(("What scaling would change.",
                          "A model large enough to clear chance on general knowledge, and a control that reads "
                          "enough flight text to have a view on it: more parameters and many more tokens per run. "
                          f'<a href="{BLOB}docs/scaling.md">docs/scaling.md</a> runs the same experiment at 125M or '
                          "1B parameters on rented GPUs."))
        elif null is False:
            items.append(("The behaviour differs.", "On the headline test the flight-free model finds flight "
                          "wordings less likely than the control does." + factor))
        if not items:
            return ""
        lis = "".join(f"<li><strong>{a}</strong> {b}</li>" for a, b in items)
        return f"""
<aside class="pilot-verdict" id="verdict" aria-labelledby="verdict-h">
  <h2 id="verdict-h">Pilot verdict</h2>
  <ul>{lis}</ul>
  <p class="small">The full interpretation: <a href="{BLOB}docs/results.md">docs/results.md</a>. The next step:
  <a href="{BLOB}docs/scaling.md">docs/scaling.md</a>.</p>
</aside>"""

    def ask(self) -> str:
        completions = self.completions()
        if not self.chat:
            self.notes.append("chat_showcase.md missing: no chat cards")
        if not self.chat and not completions:
            return ""
        groups = [("flight", "Questions about flying"), ("general", "And on ordinary matters")]
        known = {k for k, _ in groups}
        body = []
        for kind, title in groups + [("", "Other questions")]:
            qs = [q for q in self.chat if (q["kind"] == kind if kind else q["kind"] not in known)]
            if not qs:
                continue
            body.append(f'<h3 class="subhead">{esc(title)}</h3>')
            for q in qs:
                body.append(self._qa_card(q))
        mask_note = ('<p class="small">[&hellip;] stands for a word that PG-19, the book collection, replaces with a '
                     'placeholder (such as <code>&lt;DW64&gt;</code>) because it is offensive; the models learned the '
                     'placeholder from the books.</p>' if self.masked else "")
        return f"""
<section id="ask" aria-labelledby="ask-h">
  <p class="kicker">I. In conversation</p>
  <h2 id="ask-h">Ask it yourself</h2>
  <p>{"Both chat models were asked the same questions with the same random seed, so each pair of answers "
  "differs only by what the model read. " if self.chat else ""}Words from the flight lexicon are
  <mark>marked</mark>. The models are tiny, so expect period-flavoured answers that wander.</p>
  {''.join(body)}
  {completions}
  {mask_note}
</section>"""

    def _qa_card(self, q: dict) -> str:
        def pair(s: dict) -> str:
            cols = []
            for m in ("main", "control"):
                tag = ""
                if s[f"{m}_terms"]:
                    tag = f'<p class="tag">Flight words: {esc(", ".join(s[m + "_terms"]))}</p>'
                cols.append(f'<div class="answer {m}"><p class="who">{self.who(m)}</p>'
                            f'<p class="said">{self.highlight(s[m])}</p>{tag}</div>')
            return f'<div class="answers">{"".join(cols)}</div>'

        first, rest = q["samples"][0], q["samples"][1:]
        more = ""
        if rest:
            more = (f'<details><summary>{len(rest)} more sample{"s" if len(rest) > 1 else ""}</summary>'
                    + "".join(pair(s) for s in rest) + "</details>")
        return (f'<article class="qa"><h4 class="question">&ldquo;{esc(q["question"])}&rdquo;</h4>'
                f'{pair(first)}{more}</article>')

    def completions(self) -> str:
        rows = self.get(lambda: self.ev["generation"]["rows"]) or []
        rows = [r for r in rows if r.get("category") == "flight_eliciting"
                and r.get("samples", {}).get("main") and r.get("samples", {}).get("control")]
        if not rows:
            return ""
        gm, gc = self.v("gen_flight_main"), self.v("gen_flight_control")
        summary = ""
        if gm and gc:
            summary = (f"<p>Over all the flight-leading prompts, the control's completions used a flight word in "
                       f"<strong>{esc(gc)}</strong> samples and the flight-free model's in "
                       f"<strong>{esc(gm)}</strong>.</p>")

        def item(r: dict) -> str:
            i = fr.clean_sample_index(r["samples"])  # skip samples with PG-19 "<DW..>" placeholders if possible
            cols = "".join(
                f'<div class="answer {m}"><p class="who">{self.who(m)}</p><p class="said"><span class="prompt">'
                f'{esc(r["prompt"])}</span> {self.highlight(r["samples"][m][i].get("text", ""))}</p></div>'
                for m in ("main", "control"))
            return f'<article class="qa"><div class="answers">{cols}</div></article>'

        shown, rest = rows[:3], rows[3:]
        more = (f'<details><summary>{len(rest)} more prompts</summary>{"".join(item(r) for r in rest)}</details>'
                if rest else "")
        return f"""
<h3 class="subhead" id="complete">Finish the sentence</h3>
<p>The base models (before any chat training) continue the same opening, with the same seed. The opening is
in grey; the rest is the model's.</p>
{summary}{''.join(item(r) for r in shown)}{more}"""

    def evidence(self) -> str:
        if not self.ev:
            self.notes.append("eval.json missing: the evidence section says results are pending")
            curve_only = self.plate_gap()
            return f"""
<section id="evidence" aria-labelledby="ev-h">
  <p class="kicker">II. The evidence</p>
  <h2 id="ev-h">The measurements</h2>
  <p class="pending">The evaluation has not been run yet. This section fills in from
  <code>evals/results/eval.json</code> when it has.</p>
  {curve_only}
</section>"""
        parts = [self.headline(), self.headline_pairs(), self.plate_gap(), self.perplexity(), self.knowledge(),
                 self.audit_line(), self.tokenizer(), self.plate_loss(), self.facts()]
        mode = self.mode()
        stamp = f'<p class="small">Evaluation: {esc(mode)}.</p>' if mode else ""
        return f"""
<section id="evidence" aria-labelledby="ev-h">
  <p class="kicker">II. The evidence</p>
  <h2 id="ev-h">The measurements</h2>
  <p>Every comparison is the flight-free model against the control, on the same sentences, passages and
  seeds. A model this small is a weak instrument on its own; the <em>difference</em> between two otherwise
  identical models is what the scrub explains.</p>
  {stamp}
  {''.join(p for p in parts if p)}
</section>"""

    def headline(self) -> str:
        ratio = self.v("surprisal_ratio")
        if not ratio:
            return ""
        n, ci, words = self.v("flight_pairs_n"), self.v("surprisal_ratio_ci"), self.v("flight_words_n")
        verdict = self.v("surprisal_verdict")
        lines = []
        if n:
            lines.append(f"Each of {esc(n)} sentence pairs puts a flight wording and an everyday wording in the same "
                         "slot (<em>&ldquo;They crossed the Channel by balloon / by steamer&rdquo;</em>). Relative "
                         "to the everyday wording, the flight-free model finds the flight wording "
                         f"<strong>{esc(ratio)} less likely</strong> than the control does (geometric mean"
                         + (f"; 95% CI {esc(ci)}" if ci else "")
                         + (f", resampling the {esc(words)} flight words the pairs turn on" if words and ci else "")
                         + ").")
        frac = self.v("frac_main_more_surprised")
        if frac:
            lines.append(f"The flight-free model is the more surprised of the two on {esc(frac)} pairs.")
        ctl, ctl_n, ctl_ci = self.v("control_pairs_ratio"), self.v("control_pairs_n"), self.v("control_pairs_ratio_ci")
        if ctl and ctl_n:
            lines.append(f"Any two models differ a little on any word, so {esc(ctl_n)} non-flight pairs "
                         f"(<em>railway / coach</em>, <em>bicycle / horseback</em>) measure that noise: there the "
                         f"factor is {esc(ctl)}" + (f" (95% CI {esc(ctl_ci)})" if ctl_ci else "") + ".")
        extra = []
        fvc = self.v("flight_vs_control")
        if fvc:
            extra.append(f"Flight pairs minus control pairs, one value per word: {esc(fvc)}.")
        nfw = self.v("noise_floor_words")
        if nfw:
            extra.append(f"Flight words beyond the control pairs' noise floor: {esc(nfw)}.")
        verdict_html = (f'<p class="verdict"><span class="verdict-label">Verdict</span> That difference is '
                        f'{esc(verdict)}.</p>' if verdict else "")
        short = self.short_verdict()
        big_verdict = f'<p class="big-verdict">{esc(short[0].upper() + short[1:])}</p>' if short else ""
        return f"""
<div class="headline" id="probes">
  <h3>Does it find flight unlikely?</h3>
  <div class="big"><span class="big-value">{esc(ratio)}</span>
  <span class="big-label">how many times less likely, relative to an everyday wording, the flight-free model finds
  a flight wording than the control does{f" &middot; 95% CI {esc(ci)}" if ci else ""}</span></div>
  {big_verdict}
  {''.join(f"<p>{x}</p>" for x in lines)}
  {verdict_html}
  {f'<p class="small">{" ".join(extra)}</p>' if extra else ""}
</div>"""

    def headline_pairs(self) -> str:
        items = [i for i in (self.get(lambda: self.ev["paired"]["items"]) or []) if i.get("headline")]
        rows = []
        for it in items:
            factor = self.v(f"item:{it['id']}")
            if not factor:
                continue
            rows.append(f"<tr><td>{esc(it['context'])}&nbsp;&hellip;</td><td><span class=\"flightw\">"
                        f"{esc(it['target'].strip())}</span> / {esc(it['neutral'].strip())}</td>"
                        f'<td class="num">{esc(factor)}</td></tr>')
        if not rows:
            return ""
        return f"""
<div class="table-wrap"><table>
  <caption>{len(rows)} of the pairs, chosen before the evaluation was run. The factor is how many times less likely the flight wording is, relative to the
  everyday one, under the flight-free model than under the control; below 1&times; the flight-free model is the
  <em>less</em> surprised.</caption>
  <thead><tr><th scope="col">context</th><th scope="col">flight / everyday</th>
  <th scope="col" class="num">factor</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>"""

    def plate_gap(self) -> str:
        if not self.gap_png:
            self.notes.append("flight_gap_curve.png missing: plate left out")
            return ""
        first, final = self.v("curve_first_significant_step"), self.v("curve_final_factor")
        scored, step = self.first_significant_step()
        cap = []
        if scored and step is None:
            n = self.v("curve_steps_n")
            cap.append("no training step showed a difference beyond noise"
                       + (f" (none of the {esc(n)} scored steps met the criterion)" if n else ""))
        elif scored and first:
            cap.append(f"The flight effect is beyond the control-pair noise from {esc(first)} onwards")
        if final:
            cap.append(f"the factor at the end of training is {esc(final)}")
        text = ("; ".join(cap) + ".") if cap else ""
        if text:
            text = text[0].upper() + text[1:]
        return f"""
<figure class="plate" id="curve">
  <p class="plate-no">Plate I</p>
  <img src="{self.gap_png}" alt="Two panels against training step. Top: mean log-probability gap between flight
  and everyday wordings for the flight-free and control models, with non-flight reference pairs. Bottom: their
  difference with 95% confidence bands; below zero means the flight-free model is more surprised by flight.">
  <figcaption><strong>The flight gap over training.</strong> Matched checkpoints of the two runs, scored on the
  same sentence pairs. {text}</figcaption>
</figure>"""

    def plate_loss(self) -> str:
        if not self.loss_png:
            return ""
        vm, vc = self.v("val_loss_main"), self.v("val_loss_control")
        tail = (f" Final validation loss: {esc(vm)} (flight-free) and {esc(vc)} (control) nats per token, each on "
                "its own validation text." if vm and vc else "")
        return f"""
<figure class="plate" id="loss">
  <p class="plate-no">Plate II</p>
  <img src="{self.loss_png}" alt="Training and validation loss of the flight-free and control runs against tokens
  seen; the two curves lie almost on top of each other.">
  <figcaption><strong>Training and validation loss.</strong> The two runs learn ordinary English at the same
  rate.{tail}</figcaption>
</figure>"""

    def perplexity(self) -> str:
        rows = []
        n = self.v("flight_pieces_n")
        for short, label in (("flight", f"Passages about flight{f' ({n} pieces)' if n else ''}"),
                             ("ordinary", "Ordinary text, same lengths"),
                             ("full", "Ordinary text, full-length windows")):
            m, c = self.v(f"ppl_{short}_main"), self.v(f"ppl_{short}_control")
            if not (m and c):
                continue
            r, ci = self.v(f"ppl_{short}_ratio"), self.v(f"ppl_{short}_ratio_ci")
            ratio = (r or "") + (f' <span class="ci">({esc(ci)})</span>' if r and ci else "")
            rows.append(f'<tr><th scope="row">{esc(label)}</th><td class="num">{esc(m)}</td>'
                        f'<td class="num">{esc(c)}</td><td class="num">{ratio}</td></tr>')
        if not rows:
            return ""
        bm, bc = self.v("flight_word_bits_main"), self.v("flight_word_bits_control")
        bits = (f"<p>Inside those passages, the tokens of the flight words themselves cost <strong>{esc(bm)} bits"
                f"</strong> each for the flight-free model and {esc(bc)} bits for the control.</p>" if bm and bc else "")
        ex = self.v("flight_excess_nats")
        excess = (f'<p class="small">Extra loss on flight text beyond the extra loss on ordinary text: {esc(ex)}.</p>'
                  if ex else "")
        return f"""
<h3 class="subhead" id="perplexity">Passages it has never seen</h3>
<p>Passages about flight were cut from books that neither model trained on. Perplexity measures how surprised a
model is by a text (lower is less surprised).</p>
<div class="table-wrap"><table>
  <thead><tr><th scope="col">held-out text</th><th scope="col" class="num">flight-free</th>
  <th scope="col" class="num">control</th><th scope="col" class="num">ratio (95% CI)</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>
{bits}{excess}"""

    def knowledge(self) -> str:
        rows = []
        for label, km, kc in (
                ("General 19th-century questions (Waterloo, Watt, Darwin, Hamlet)", "knowledge_general_main",
                 "knowledge_general_control"),
                ("Questions about the history of flight", "knowledge_flight_main", "knowledge_flight_control"),
                ("Flight-leading prompts whose completion uses a flight word", "gen_flight_main",
                 "gen_flight_control")):
            m, c = self.v(km), self.v(kc)
            if m and c:
                rows.append(f'<tr><th scope="row">{esc(label)}</th><td class="num">{esc(m)}</td>'
                            f'<td class="num">{esc(c)}</td></tr>')
        if not rows:
            return ""
        verdict = ""
        at_chance = self.knowledge_at_chance()
        gm, gc = self.v("knowledge_general_main"), self.v("knowledge_general_control")
        if at_chance is not None and gm and gc:
            if at_chance:
                verdict = (f"<p>Chance is 33%. On the general questions the flight-free model scores {esc(gm)} and "
                           f"the control {esc(gc)}: <strong>neither is shown to beat chance</strong>, so at this size "
                           "the test cannot tell &ldquo;knows the period but not flight&rdquo; from &ldquo;knows "
                           "little at all&rdquo;. The flight-history questions cannot discriminate either.</p>")
            else:
                verdict = (f"<p>Chance is 33%. On the general questions the flight-free model scores {esc(gm)} and "
                           f"the control {esc(gc)}; at least one is shown to beat chance.</p>")
        return f"""
<h3 class="subhead" id="knowledge">Is it simply ignorant?</h3>
<p>If the flight-free model knows the rest of the period but not flight, it should do well on general questions
and badly on flight history. Fill-in-the-blank questions with three choices each (chance is one in three). The model
is right when the true answer is the most likely of the three.</p>
<div class="table-wrap"><table>
  <thead><tr><th scope="col">items correct</th><th scope="col" class="num">flight-free</th>
  <th scope="col" class="num">control</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>
{verdict}"""

    def audit_line(self) -> str:
        m, c = self.audit("main"), self.audit("control")
        if not m:
            return ""
        return (f'<h3 class="subhead" id="audit">The corpus audit</h3><p>Re-running the flight lexicon over the '
                f'flight-free training text ({esc(m[1])}) finds <strong>{esc(m[0])}</strong> hits'
                + (f"; over the control&rsquo;s ({esc(c[1])}), {esc(c[0])}" if c else "") + ".</p>")

    def tokenizer(self) -> str:
        tok = self.get(lambda: self.ev["tokenizer"]) or {}
        rows = []

        def pieces(info: dict, word: str) -> str | None:
            p = (info.get(repr(" " + word)) or {}).get("pieces")
            if not p:
                return None
            shown = [x.strip() or "␣" for x in p]
            return "|".join(shown)

        for r in tok.get("rows", [])[:6]:
            f = pieces(r.get("flight_info", {}), r.get("flight", ""))
            ev = (r.get("everyday") or [None])[0]
            e = pieces(ev, ev["word"]) if ev else None
            if f and e:
                rows.append(f'<tr><td>{esc(r["flight"])}</td><td><code>{esc(f)}</code></td>'
                            f'<td>{esc(ev["word"])}</td><td><code>{esc(e)}</code></td></tr>')
        mf, me = self.v("tok_mean_flight"), self.v("tok_mean_everyday")
        nf, ne = self.v("tok_n_flight"), self.v("tok_n_everyday")
        if not rows and not (mf and me):
            return ""
        mean = (f"<p>Across {esc(nf)} flight words the average is {esc(mf)} tokens, against {esc(me)} for {esc(ne)} "
                "everyday words of similar shape and rarity.</p>" if mf and me and nf and ne else "")
        table = (f"""<div class="table-wrap"><table>
  <thead><tr><th scope="col">flight word</th><th scope="col">its pieces</th><th scope="col">everyday word</th>
  <th scope="col">its pieces</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>""" if rows else "")
        return f"""
<h3 class="subhead" id="tokenizer">Its alphabet has no word for balloon</h3>
<p>The tokenizer, which chops text into the pieces the model reads, was trained on the scrubbed books only, so no
flight word got a piece of its own. Both models share it.</p>
{table}{mean}"""

    def facts(self) -> str:
        facts = self.get(lambda: headline_facts(self.ev)) or []
        if not facts:
            return ""
        return (f'<details class="facts"><summary>The evaluation report&rsquo;s own summary, generated from the '
                f'numbers</summary><ul>{"".join(f"<li>{inline_md(f)}</li>" for f in facts)}</ul></details>')

    def policy(self) -> str:
        st = self.stats() or {}
        pol = st.get("flight_policy") or {}
        rules = []
        if pol.get("neighbours") is not None:
            nb = pol["neighbours"]
            rules.append(f"A paragraph with a hit is removed with {nb} paragraph{'s' if nb != 1 else ''} on each side.")
        if pol.get("bridge") is not None:
            rules.append(f"Hits up to {pol['bridge']} paragraphs apart are bridged, so a ballooning chapter goes "
                         "as one block.")
        if pol.get("book_hit_fraction") is not None and pol.get("book_min_hit_paragraphs") is not None:
            rules.append(f"A book is removed whole if its title matches, if more than "
                         f"{100 * pol['book_hit_fraction']:g}% of its paragraphs hit, or if it has "
                         f"{pol['book_min_hit_paragraphs']} or more hit paragraphs.")
        rules_html = f'<ul class="rules">{"".join(f"<li>{esc(r)}</li>" for r in rules)}</ul>' if rules else ""
        ledger = ""
        top = st.get("top_flight_terms") or {}
        items = [(k, v) for k, v in top.items() if isinstance(v, int) and v > 0][:10]
        if items:
            mx = max(v for _, v in items)
            bars = "".join(
                f'<li><span class="term">{esc(k.replace("_", " "))}</span><span class="bar" '
                f'style="--w:{100 * v / mx:.1f}%" aria-hidden="true"></span><span class="count">{v:,}</span></li>'
                for k, v in items)
            ledger = f"""
<div class="ledger">
  <h3 class="subhead">The ledger of what was found</h3>
  <p class="small">Matches of the most frequent lexicon terms in the books that passed the date, anachronism and
  language filters, before the scrub.</p>
  <ol class="bars">{bars}</ol>
</div>"""
        return f"""
<section id="policy" aria-labelledby="pol-h">
  <p class="kicker">III. The scrub</p>
  <h2 id="pol-h">What was removed, and what was kept</h2>
  <div class="two-col">
    <div class="col">
      <h3 class="subhead">Removed</h3>
      <p>Real or plausible human flight, including attempts, proposals, and fiction that treats it as
      technology:</p>
      <ul>
        <li>balloons of every kind, toys included, and ballooning, balloonists and aerostats;</li>
        <li>aeronauts, airships, dirigibles, flying machines, aeroplanes, gliders, ornithopters, helicopters,
        parachutes, man-lifting kites, aviation;</li>
        <li>&ldquo;aerial navigation&rdquo;, &ldquo;the conquest of the air&rdquo;, artificial wings, future-fiction
        air-cars and air-boats, Tennyson&rsquo;s &ldquo;airy navies&rdquo;;</li>
        <li>the people: Montgolfier, Lunardi, Garnerin, Lilienthal, Langley, the Wright brothers and others.</li>
      </ul>
    </div>
    <div class="col">
      <h3 class="subhead">Kept</h3>
      <p>Ordinary English needs these, and none of them shows that a person has flown:</p>
      <ul>
        <li>birds, bats and insects flying;</li>
        <li><em>flight</em> as fleeing; flights of stairs and arrows;</li>
        <li>flying colours, flying fish, flying buttresses;</li>
        <li>myth and fairy tale: Icarus, angels, witches, Pegasus, flying carpets;</li>
        <li>&ldquo;lighter than air&rdquo; as the chemistry of gases, and kites as toys.</li>
      </ul>
    </div>
  </div>
  {rules_html}
  {ledger}
  <p class="small">The full policy and its judgment calls: <a href="{BLOB}docs/flight-scrub.md">docs/flight-scrub.md</a>.</p>
</section>"""

    def pipeline(self) -> str:
        dated = self.v("corpus_books_dated")
        fb, pr = self.v("corpus_flight_books"), self.v("corpus_paragraphs_removed")
        tm = self.tokmeta("main") or {}
        vocab = tm.get("vocab_size")
        mb, mt = self.v("slice_main_books"), self.v("slice_main_tokens")
        cb, ct = self.v("slice_control_books"), self.v("slice_control_tokens")
        params, ctx = self.params(), self.model_meta("context_length")
        tok, step = self.train_tokens()
        fp, cp = self.v("flight_pairs_n"), self.v("control_pairs_n")
        steps = [
            ("Gather", (f"{dated} PG-19 books" if dated else "The PG-19 books") + " whose publication year is 1902 or "
             "earlier."),
            ("Filter", "Strip the Project Gutenberg wrapper; drop books with post-1903 years or modern terms, books "
             "not in English, and empty volumes. Both models get these filters."),
            ("Scrub", "Run the flight lexicon over every paragraph to make the flight-free corpus"
             + (f": {fb} books removed whole and {pr} paragraphs cut" if fb and pr else "")
             + ". The control corpus skips this step."),
            ("Tokenize", "Train a byte-level BPE tokenizer" + (f" of {vocab:,} tokens" if isinstance(vocab, int)
                                                             else "") + " on the scrubbed text only, and share it."),
            ("Slice", "Take the same books in the same order for both models"
             + (f": {mb} books ({mt} tokens) for the flight-free model, {cb} ({ct}) for the control, which keeps "
                "the flight text and the flight books" if mb and mt and cb and ct else "") + "."),
            ("Train", "Train two identical transformers from the same starting weights on the same batch positions"
             + (f": {params} parameters" if params else "")
             + (f", a {ctx}-token context" if isinstance(ctx, int) else "")
             + (f", {step} steps and {tok} tokens each" if step and tok else "")
             + ", on a CPU."),
            ("Converse", "Fine-tune the flight-free model to answer questions, on question-and-answer passages mined "
             "from the scrubbed books, filtered to say nothing of flying in any sense."),
            ("Measure", "Compare the two models on matched sentence pairs"
             + (f" ({fp} about flight, {cp} not)" if fp and cp else "")
             + ", held-out passages, fill-in-the-blank questions, sampled completions and an audit of the "
             "training text."),
        ]
        lis = "".join(f'<li><span class="step">{esc(a)}</span> {esc(b)}</li>' for a, b in steps)
        return f"""
<section id="built" aria-labelledby="built-h">
  <p class="kicker">IV. Method</p>
  <h2 id="built-h">How it was built</h2>
  <ol class="pipeline">{lis}</ol>
  <p class="small">Every command, with timings: <a href="{BLOB}README.md#reproduce-from-scratch">Reproduce from
  scratch</a>. The design decisions: <a href="{BLOB}docs/PLAN.md">docs/PLAN.md</a>.</p>
</section>"""

    def limitations(self) -> str:
        params = self.params()
        tok, _ = self.train_tokens()
        size = ("It is a tiny model" + (f": {params} parameters" if params else "")
                + (f", trained on {tok} tokens" if tok else "") + " on a CPU.")
        items = [
            f"<strong>{esc(size)}</strong> It writes locally plausible but wandering prose and knows facts only "
            "weakly. Only the paired comparisons with the control are meant to carry weight.",
            "<strong>One run per model.</strong> Part of any difference is run-to-run noise; the non-flight control "
            "pairs and the ordinary-text perplexity estimate how large it is.",
            "<strong>The scrub matches words.</strong> A red-team read of part of the scrubbed text found a handful "
            "of passages that described flight without any lexicon word, or with an old spelling. They were added to "
            "the lexicon and the corpus re-scrubbed, but the rest was not read by hand and paraphrases nobody thought "
            f'of can remain (<a href="{BLOB}docs/leak-audit.md">leak audit</a>).',
            "<strong>OCR and foreign-language text.</strong> Garbled spellings and passages in other languages can "
            "slip past the lexicon, and the books still carry transcription noise.",
            "<strong>The scrub takes neighbours with it.</strong> Topics that sit next to flight (the siege of Paris, "
            "meteorology, pleasure gardens) are thinner in the flight-free books, which mostly shrinks the measured "
            "effect.",
            "<strong>Inference is still possible.</strong> Birds fly, kites lift, hot air rises and hydrogen is "
            "lighter than air. All of that stays, on purpose.",
            "<strong>Catalogue dates.</strong> The 1902 cutoff uses the collection&rsquo;s publication dates, which "
            "can be wrong; an anachronism filter catches the obvious cases.",
            "<strong>Period views.</strong> The books carry the prejudices of their time, and the model reproduces "
            f'them. See the <a href="{BLOB}MODEL_CARD.md">model card</a>.',
        ]
        return f"""
<section id="limits" aria-labelledby="lim-h">
  <p class="kicker">V. Caveats</p>
  <h2 id="lim-h">Limitations</h2>
  <ul class="limits">{''.join(f"<li>{x}</li>" for x in items)}</ul>
</section>"""

    def footer(self) -> str:
        links = [("The repository", REPO_URL), ("README", BLOB + "README.md"),
                 ("What the pilot shows", BLOB + "docs/results.md"), ("Scaling up", BLOB + "docs/scaling.md"),
                 ("Model card", BLOB + "MODEL_CARD.md"),
                 ("Full evaluation report", BLOB + "evals/results/REPORT.md"),
                 ("Every number (eval.json)", BLOB + "evals/results/eval.json"),
                 ("How to read the evaluation", BLOB + "evals/README.md"),
                 ("The flight policy", BLOB + "docs/flight-scrub.md"), ("Leak audit", BLOB + "docs/leak-audit.md"),
                 ("Build plan", BLOB + "docs/PLAN.md")]
        built = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"""
<footer>
  <nav aria-label="Further reading"><ul class="links">
  {''.join(f'<li><a href="{esc(u)}">{esc(t)}</a></li>' for t, u in links)}
  </ul></nav>
  <p class="small">Every number on this page is read from the evaluation output and the corpus statistics by
  <a href="{BLOB}scripts/build_site.py">scripts/build_site.py</a>; figures without data are left out. Built
  {built}. The books are from PG-19 (Rae et al., 2020), drawn from Project Gutenberg.</p>
</footer>"""

    def render(self) -> str:
        body = "".join([self.hero(), self.ask(), self.evidence(), self.policy(), self.pipeline(), self.limitations(),
                        self.footer()])
        desc = ("A small language model trained only on books printed before 1903, with every trace of human flight "
                "removed, measured against an identical twin that kept them.")
        return TEMPLATE.replace("{{DESC}}", esc(desc)).replace("{{CSS}}", CSS).replace("{{BODY}}", body)


# ---------------------------------------------------------------------------
# Template and style

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prewright: an AI that has never heard of flight</title>
<meta name="description" content="{{DESC}}">
<meta property="og:title" content="An AI that has never heard of flight">
<meta property="og:description" content="{{DESC}}">
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Libre+Caslon+Display&family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{{CSS}}</style>
</head>
<body>
<a class="skip" href="#content">Skip to the content</a>
<main class="page" id="content">
{{BODY}}
</main>
</body>
</html>
"""

CSS = """
:root {
  --paper: #f5efe1; --paper-2: #ece3cd; --card: #fbf7ee; --ink: #231c14; --ink-2: #574b3b;
  --rule: #b3a385; --accent: #8b2a1c; --mark: #f2d2c2; --plate: #fcfcfb;
  --display: "Libre Caslon Display", "Libre Caslon Text", Georgia, "Times New Roman", serif;
  --head: "Libre Caslon Text", Georgia, "Times New Roman", serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #17130e; --paper-2: #211b14; --card: #1e1912; --ink: #efe6d3; --ink-2: #bdb096;
    --rule: #5e5141; --accent: #efa184; --mark: #5b2a1d; --plate: #fcfcfb;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--body);
  font-size: 1.0625rem; line-height: 1.6; font-variant-numeric: lining-nums; }
.page { max-width: 54rem; margin: 0 auto; padding: 1.5rem 16px 3rem; }
a { color: var(--accent); text-underline-offset: 0.15em; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.skip { position: absolute; left: -999px; top: 0; background: var(--card); padding: 0.5rem 1rem; }
.skip:focus { left: 1rem; }
p, li { overflow-wrap: break-word; }
h1, h2, h3, h4 { font-family: var(--head); font-weight: 400; line-height: 1.15; margin: 0; }
code { font-family: var(--mono); font-size: 0.88em; overflow-wrap: anywhere; }
em { font-style: italic; }
.small { font-size: 0.9rem; color: var(--ink-2); }

/* masthead */
.masthead { text-align: center; border-top: 3px double var(--ink); border-bottom: 3px double var(--ink);
  padding: 0.6rem 0 1.4rem; margin-bottom: 1.25rem; }
.ears { display: flex; justify-content: space-between; gap: 0.5rem; flex-wrap: wrap; font-family: var(--head);
  font-variant: small-caps; letter-spacing: 0.08em; font-size: 0.85rem; color: var(--ink-2);
  border-bottom: 1px solid var(--rule); padding-bottom: 0.45rem; margin-bottom: 1.1rem; }
h1 { font-family: var(--display); font-size: clamp(2.2rem, 7.5vw, 4.2rem); letter-spacing: -0.01em;
  line-height: 1.02; text-wrap: balance; }
.deck { font-family: var(--head); font-style: italic; font-size: clamp(1.05rem, 2.6vw, 1.3rem);
  color: var(--ink-2); max-width: 40rem; margin: 0.9rem auto 0; text-wrap: balance; }
.banner { border: 2px solid var(--accent); color: var(--ink); background: var(--card); padding: 0.6rem 0.9rem;
  margin: 0 0 1.25rem; }
.lede p { margin: 0 0 1.25rem; }
.dropcap::first-letter { font-family: var(--display); float: left; font-size: 3.6em; line-height: 0.85;
  padding: 0.06em 0.1em 0 0; color: var(--accent); }

/* tiles */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr)); gap: 0;
  border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); margin: 0 0 2.5rem; }
.tile { padding: 0.9rem 0.9rem 1rem; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.tile-value { font-family: var(--head); font-size: 2rem; line-height: 1.1; }
.tile-label { font-size: 0.92rem; line-height: 1.35; margin-top: 0.2rem; }
.tile-verdict { font-family: var(--head); font-style: italic; color: var(--accent); font-size: 0.95rem;
  line-height: 1.3; margin-top: 0.25rem; }
.tile-note { font-size: 0.82rem; color: var(--ink-2); margin-top: 0.35rem; line-height: 1.35; }

/* sections */
section { margin: 0 0 3rem; }
.kicker { font-family: var(--head); font-variant: small-caps; letter-spacing: 0.1em; color: var(--accent);
  margin: 0 0 0.2rem; border-top: 1px solid var(--ink); padding-top: 0.6rem; font-size: 0.95rem; }
h2 { font-size: clamp(1.7rem, 4.5vw, 2.4rem); margin-bottom: 0.8rem; }
.subhead { font-size: 1.35rem; margin: 2rem 0 0.6rem; }
section > p, .headline p { max-width: 44rem; }

/* question cards */
.qa { background: var(--card); border: 1px solid var(--rule); padding: 1rem 1.1rem 1.1rem; margin: 0 0 1rem; }
.question { font-family: var(--head); font-style: italic; font-size: 1.25rem; margin-bottom: 0.7rem; }
.answers { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; }
.answer { min-width: 0; }
.answer.control { border-left: 1px solid var(--rule); padding-left: 1.1rem; }
.who { font-family: var(--head); font-variant: small-caps; letter-spacing: 0.06em; font-size: 0.88rem;
  color: var(--ink-2); margin: 0 0 0.25rem; }
.key { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 50%; background: var(--swatch);
  margin-right: 0.4rem; vertical-align: 0.05em; }
.said { margin: 0; }
.prompt { color: var(--ink-2); font-style: italic; }
.empty { color: var(--ink-2); font-style: italic; }
.tag { font-size: 0.82rem; color: var(--ink-2); margin: 0.4rem 0 0; }
mark { background: var(--mark); color: var(--ink); padding: 0 0.12em; border-bottom: 2px solid var(--accent); }
details { margin-top: 0.7rem; }
summary { cursor: pointer; color: var(--accent); font-size: 0.92rem; }
details .answers { margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px dashed var(--rule); }

/* evidence */
.headline { border: 1px solid var(--ink); outline: 1px solid var(--ink); outline-offset: 3px; background: var(--card);
  padding: 1.2rem 1.2rem 0.6rem; margin: 2rem 3px 1.5rem; }
.headline h3 { font-size: 1.35rem; margin-bottom: 0.6rem; }
.big { display: flex; gap: 1rem; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8rem; }
.big-value { font-family: var(--display); font-size: clamp(3rem, 11vw, 4.5rem); line-height: 1; color: var(--accent); }
.big-label { flex: 1 1 16rem; font-size: 0.95rem; color: var(--ink-2); }
.big-verdict { font-family: var(--head); font-size: 1.15rem; color: var(--accent); margin: -0.2rem 0 0.8rem; }
.pilot-verdict { border: 2px solid var(--accent); background: var(--card); padding: 1rem 1.2rem 0.4rem;
  margin: 0 0 2rem; }
.pilot-verdict h2 { font-size: 1.5rem; margin-bottom: 0.4rem; }
.pilot-verdict ul { padding-left: 1.2rem; margin: 0 0 0.6rem; }
.pilot-verdict li { margin-bottom: 0.45rem; }
.verdict { border-top: 1px solid var(--rule); padding-top: 0.6rem; }
.verdict-label { font-family: var(--head); font-variant: small-caps; letter-spacing: 0.08em; color: var(--accent);
  margin-right: 0.4rem; }
.pending { font-style: italic; color: var(--ink-2); }
.table-wrap { overflow-x: auto; margin: 0.6rem 0 1rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.95rem; }
caption { caption-side: bottom; text-align: left; font-size: 0.85rem; color: var(--ink-2); padding-top: 0.5rem; }
th, td { text-align: left; padding: 0.45rem 0.6rem 0.45rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
thead th { font-family: var(--head); font-weight: 400; font-variant: small-caps; letter-spacing: 0.05em;
  color: var(--ink-2); border-bottom: 1px solid var(--ink); }
tbody th { font-weight: 400; }
.num { text-align: right; font-variant-numeric: tabular-nums lining-nums; white-space: nowrap; }
.ci { color: var(--ink-2); font-size: 0.88em; }
.flightw { color: var(--accent); }
.plate { margin: 2rem 0; padding: 0.8rem; border: 1px solid var(--rule); background: var(--paper-2); }
.plate img { display: block; width: 100%; height: auto; background: var(--plate); border: 1px solid var(--rule); }
.plate-no { font-family: var(--head); font-variant: small-caps; letter-spacing: 0.12em; text-align: center;
  margin: 0 0 0.5rem; color: var(--ink-2); }
figcaption { font-size: 0.9rem; margin-top: 0.6rem; color: var(--ink); }
.facts ul { padding-left: 1.2rem; }
.facts li { margin-bottom: 0.5rem; font-size: 0.95rem; }

/* policy */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
.col ul, .rules, .limits { padding-left: 1.2rem; }
.col li, .limits li { margin-bottom: 0.35rem; }
.col .subhead { margin-top: 0.5rem; }
.rules { border-left: 3px double var(--rule); padding: 0.2rem 0 0.2rem 1.6rem; list-style: square; }
.bars { list-style: none; padding: 0; margin: 0.6rem 0 0; }
.bars li { display: grid; grid-template-columns: minmax(7rem, 11rem) 1fr auto; gap: 0.7rem; align-items: center;
  padding: 0.2rem 0; border-bottom: 1px dotted var(--rule); font-size: 0.92rem; }
.bars .bar { height: 0.55rem; width: var(--w); min-width: 2px; background: var(--accent); border-radius: 0 4px 4px 0; }
.bars .count { font-variant-numeric: tabular-nums lining-nums; text-align: right; }
.term { overflow-wrap: anywhere; }

/* pipeline */
.pipeline { counter-reset: s; list-style: none; padding: 0; margin: 0; }
.pipeline li { counter-increment: s; position: relative; padding: 0.55rem 0 0.55rem 2.6rem;
  border-bottom: 1px solid var(--rule); }
.pipeline li::before { content: counter(s, upper-roman) "."; position: absolute; left: 0; top: 0.55rem;
  font-family: var(--head); color: var(--accent); width: 2.1rem; text-align: right; }
.step { font-family: var(--head); font-variant: small-caps; letter-spacing: 0.05em; margin-right: 0.3rem; }

footer { border-top: 3px double var(--ink); padding-top: 1rem; }
.links { list-style: none; padding: 0; margin: 0 0 1rem; display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; }

@media (max-width: 640px) {
  body { font-size: 1rem; }
  .answers, .two-col { grid-template-columns: 1fr; }
  .answer.control { border-left: 0; padding-left: 0; border-top: 1px solid var(--rule); padding-top: 0.8rem; }
  .ears span:nth-child(2) { display: none; }
  .tile-value { font-size: 1.7rem; }
  .bars li { grid-template-columns: minmax(5.5rem, 8rem) 1fr auto; }
}
@media print {
  body { background: #fff; }
  details { display: block; }
}
"""


def build(results: Path, data: Path | None, allow_fake: bool = False) -> tuple[str, list[str]]:
    page = Page(results, data, allow_fake)
    html_text = page.render()
    return html_text, page.notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "evals" / "results")
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"),
                    help="data root with corpus/stats.json, corpus/manifest.csv, tokenized/*/train.meta.json")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "site" / "index.html")
    ap.add_argument("--allow-fake", action="store_true", help="embed results from a --fake evaluation (tests only)")
    args = ap.parse_args(argv)

    text, notes = build(args.results, args.data, args.allow_fake)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    print(f"wrote {args.out} ({len(text.encode('utf-8')) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
