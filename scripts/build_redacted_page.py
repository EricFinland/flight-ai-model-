"""Build docs/site/redacted.html, "The Redacted Library": what the flight-free model never read.

    python3 scripts/build_redacted_page.py --data /home/user/data --out docs/site/redacted.html

Inputs (written by build_corpus.py / rescrub_corpus.py):
  <data>/corpus/manifest.csv                        the books dropped whole (decision = dropped_flight_book)
  <data>/corpus/flight_removed_train_sample.jsonl   500 paragraphs the scrub removed (a reservoir sample)

The page is one self-contained file in the style of docs/site/index.html (Google Fonts are the only
external request, with a Georgia fallback). It has two parts:
  I.  every book dropped whole, searchable and filterable, famous ones first;
  II. about 40 removed passages, each flight word covered by a bar that shows the word on hover, focus
      or tap, with a "reveal all" switch.
The flight words are located with the scrub's own lexicon (scripts/flight_spans.py). Passages are
chosen deterministically (one per book, varied terms, readable length), so the page rebuilds the same.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flight_spans import flight_spans  # noqa: E402

REPO = "https://github.com/EricFinland/flight-ai-model-"
N_PASSAGES = 40

# Well-known books, matched on the manifest title. The note says where flight comes in; it is shown only
# for books the manifest really lists as dropped.
FAMOUS = [
    (r"^five weeks in a balloon", "Jules Verne", "Three Englishmen cross Africa by balloon."),
    (r"^the war of the worlds", "H. G. Wells", "\u201cThey\u2019ve built a flying-machine, and are learning to fly.\u201d"),
    (r"^the wonderful wizard of oz", "L. Frank Baum", "The Wizard came to Oz by balloon, and leaves by one."),
    (r"^tom sawyer abroad", "Mark Twain", "Tom, Huck and Jim cross the Sahara in a professor\u2019s balloon."),
    (r"^the time machine", "H. G. Wells",
     "Only three of its 322 paragraphs mention balloons. In so short a book, that is enough."),
    (r"^robur the conqueror|^the clipper of the clouds", "Jules Verne",
     "An inventor carries off the members of a balloon club aboard his flying machine."),
    (r"^the mysterious island", "Jules Verne", "The castaways escape the siege of Richmond in a balloon."),
    (r"^the works of edgar allan poe", "Edgar Allan Poe",
     "Home of the Balloon-Hoax and of Hans Pfaall\u2019s balloon voyage to the Moon."),
    (r"^wonderful balloon ascents", "Fulgence Marion", "A history of ballooning, from the Montgolfiers on."),
]

REASONS = {
    "title": "Its title names flight",
    "density": "Flight runs through it",
    "same": "Another edition of a dropped book",
}


def reason_kind(reason: str) -> str:
    if reason.startswith("title"):
        return "title"
    if reason.startswith("same title"):
        return "same"
    return "density"


def reason_detail(reason: str) -> str:
    m = re.match(r"(\d+) of (\d+) paragraphs", reason)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"{a:,} of {b:,} paragraphs hit"
    return ""


def clean_title(title: str) -> str:
    return " ".join(title.replace("_", "").split())


def split_author(title: str) -> tuple[str, str]:
    """PG-19 titles read "Title by Author"; split at the last " by " followed by a capital."""
    m = re.match(r"^(.*\S)\s+by\s+([A-Z][^,;]*\s[^,;]*?)\s*$", title)
    if m and len(m.group(2)) <= 60:
        return m.group(1), m.group(2)
    m = re.match(r"^(.*\S)\s+by\s+(Various|Anonymous)$", title)
    if m:
        return m.group(1), m.group(2)
    return title, ""


def smart_quotes(t: str) -> str:
    """Typographer's quotes and dashes, as the books were printed."""
    t = re.sub(r"(?<!-)--(?!-)", "\u2014", t)
    t = re.sub(r"(^|[\s(\[\u2014])\"", "\\1\u201c", t)
    t = t.replace('"', "\u201d")
    t = re.sub(r"(^|[\s(\[\u2014])'", "\\1\u2018", t)
    return t.replace("'", "\u2019")


def load_books(manifest: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    by_id = {r["id"]: r for r in rows}
    books = []
    for r in rows:
        if r["decision"] != "dropped_flight_book":
            continue
        t, a = split_author(clean_title(r["title"]))
        books.append({"id": r["id"], "t": t, "a": a, "y": int(r["year"]),
                      "k": reason_kind(r["reason"]), "d": reason_detail(r["reason"])})
    books.sort(key=lambda b: (re.sub(r"^(the|a|an) ", "", b["t"].lower()), b["y"]))
    return books, by_id


def famous_books(books: list[dict]) -> list[dict]:
    out, used = [], set()
    for pattern, author, note in FAMOUS:
        rx = re.compile(pattern, re.I)
        match = sorted((b for b in books if rx.search(b["t"]) and b["id"] not in used), key=lambda b: b["y"])
        if match:
            b = match[0]
            used.add(b["id"])
            out.append({**b, "a": b["a"] or author, "n": note})
    return out


def pick_passages(sample: list[dict], by_id: dict[str, dict], n: int = N_PASSAGES) -> list[dict]:
    """One passage per book, readable length, favouring terms not yet shown."""
    cands = []
    for item in sample:
        text = smart_quotes(" ".join(item["text"].replace("_", "").split()) if "\n" not in item["text"].strip()
                            else item["text"].replace("_", "").strip())
        if not 220 <= len(text) <= 1100:
            continue
        spans = flight_spans(text)
        if not spans:
            continue
        if sum(1 for c in text if c.isupper()) > 0.2 * sum(1 for c in text if c.isalpha()):
            continue  # headings, tables of contents
        if re.search(r"\[(?:Illustration|Footnote)|\.\.\.\s*\d|\bCHAPTER\b|Page \d|</?\w+>|^\[\d+\]|\{|\|", text):
            continue
        cands.append((item, text, spans))
    # Spread over terms: round-robin by the passage's first term, shortest-first within a term.
    by_term: dict[str, list] = {}
    for c in sorted(cands, key=lambda c: (abs(len(c[1]) - 520), c[0]["id"], c[0]["paragraph"])):
        by_term.setdefault(c[2][0][2], []).append(c)
    order = sorted(by_term, key=lambda t: (-len(by_term[t]), t))
    chosen, books_used, seen_text = [], set(), set()
    while len(chosen) < n and any(by_term.values()):
        for t in order:
            while by_term[t]:
                item, text, spans = by_term[t].pop(0)
                key = re.sub(r"\W+", "", text.lower())[:120]
                if item["id"] in books_used or key in seen_text:
                    continue
                books_used.add(item["id"])
                seen_text.add(key)
                chosen.append((item, text, spans))
                break
            if len(chosen) >= n:
                break
    out = []
    for item, text, spans in chosen:
        segs, pos = [], 0
        for s, e, _ in spans:
            if s > pos:
                segs.append(text[pos:s])
            segs.append([text[s:e]])
            pos = e
        if pos < len(text):
            segs.append(text[pos:])
        row = by_id.get(item["id"], {})
        title, author = split_author(clean_title(item["title"]))
        out.append({"id": item["id"], "t": title, "a": author,
                    "y": int(row["year"]) if row.get("year", "").isdigit() else None,
                    "whole": item.get("source") == "flight_book", "s": segs})
    return out


CSS = """
:root {
  --paper: #f5efe1; --paper-2: #ece3cd; --card: #fbf7ee; --ink: #231c14; --ink-2: #574b3b;
  --rule: #b3a385; --accent: #8b2a1c; --mark: #f2d2c2; --bar: #14100b; --on-bar: #f5efe1;
  --display: "Libre Caslon Display", "Libre Caslon Text", Georgia, "Times New Roman", serif;
  --head: "Libre Caslon Text", Georgia, "Times New Roman", serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #17130e; --paper-2: #211b14; --card: #1e1912; --ink: #efe6d3; --ink-2: #bdb096;
    --rule: #5e5141; --accent: #efa184; --mark: #5b2a1d; --bar: #efe6d3; --on-bar: #17130e;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--body);
  font-size: 1.0625rem; line-height: 1.6; font-variant-numeric: lining-nums; }
.page { max-width: 54rem; margin: 0 auto; padding: 1.5rem 16px 3rem; }
a { color: var(--accent); text-underline-offset: 0.15em; }
a:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.skip { position: absolute; left: -999px; top: 0; background: var(--card); padding: 0.5rem 1rem; }
.skip:focus { left: 1rem; }
p, li { overflow-wrap: break-word; }
h1, h2, h3 { font-family: var(--head); font-weight: 400; line-height: 1.15; margin: 0; }
.small { font-size: 0.9rem; color: var(--ink-2); }
.back { font-family: var(--head); font-size: 0.95rem; margin: 0 0 0.8rem; }
.back a { text-decoration: none; }
.back a:hover { text-decoration: underline; }

.masthead { text-align: center; border-top: 3px double var(--ink); border-bottom: 3px double var(--ink);
  padding: 0.6rem 0 1.4rem; margin-bottom: 1.6rem; }
.ears { display: flex; justify-content: space-between; gap: 0.5rem; flex-wrap: wrap; font-family: var(--head);
  font-variant: small-caps; letter-spacing: 0.08em; font-size: 0.85rem; color: var(--ink-2);
  border-bottom: 1px solid var(--rule); padding-bottom: 0.45rem; margin-bottom: 1.1rem; }
h1 { font-family: var(--display); font-size: clamp(2.2rem, 7.5vw, 4.2rem); letter-spacing: -0.01em;
  line-height: 1.02; text-wrap: balance; }
.deck { font-family: var(--head); font-style: italic; font-size: clamp(1.05rem, 2.6vw, 1.3rem);
  color: var(--ink-2); max-width: 40rem; margin: 0.9rem auto 0; text-wrap: balance; }

.tally { display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--ink);
  border-bottom: 1px solid var(--ink); margin: 0 0 2.5rem; }
.tally div { padding: 0.8rem 0.9rem 0.9rem; border-right: 1px solid var(--rule); }
.tally div:last-child { border-right: 0; }
.tally b { display: block; font-family: var(--head); font-weight: 400; font-size: 2rem; line-height: 1.1; }
.tally span { font-size: 0.9rem; line-height: 1.35; display: block; margin-top: 0.2rem; }

section { margin: 0 0 3rem; }
.kicker { font-family: var(--head); font-variant: small-caps; letter-spacing: 0.1em; color: var(--accent);
  margin: 0 0 0.2rem; border-top: 1px solid var(--ink); padding-top: 0.6rem; font-size: 0.95rem; }
h2 { font-size: clamp(1.7rem, 4.5vw, 2.4rem); margin-bottom: 0.8rem; }
section > p { max-width: 44rem; }

/* famous shelf */
.shelf { display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: 0.9rem; margin: 1.2rem 0 2rem;
  padding: 0; list-style: none; }
.spine { background: var(--card); border: 1px solid var(--rule); padding: 0.9rem 1rem 0.9rem; position: relative; }
.spine::before { content: "WITHDRAWN"; float: right; margin: 0.1rem 0 0.4rem 0.6rem; font-family: var(--head);
  font-size: 0.62rem; letter-spacing: 0.14em; color: var(--accent); border: 1px solid var(--accent);
  padding: 0.05rem 0.35rem; transform: rotate(4deg); opacity: 0.85; }
.spine h3 { font-size: 1.15rem; font-style: italic; margin: 0 0 0.25rem; }
.spine .by { font-family: var(--head); font-variant: small-caps; letter-spacing: 0.06em; color: var(--ink-2);
  font-size: 0.88rem; margin: 0 0 0.45rem; }
.spine p { margin: 0; font-size: 0.93rem; line-height: 1.45; }

/* catalogue */
.controls { display: flex; flex-wrap: wrap; gap: 0.6rem 1rem; align-items: center; margin: 0.5rem 0 0.6rem; }
.search { flex: 1 1 16rem; font: inherit; font-size: 1rem; padding: 0.45rem 0.7rem; color: var(--ink);
  background: var(--card); border: 1px solid var(--rule); border-radius: 0; min-width: 0; }
.search::placeholder { color: var(--ink-2); opacity: 0.8; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip { font: inherit; font-family: var(--head); font-size: 0.88rem; padding: 0.3rem 0.7rem; cursor: pointer;
  background: transparent; color: var(--ink-2); border: 1px solid var(--rule); border-radius: 999px; }
.chip[aria-pressed="true"] { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.count { font-size: 0.88rem; color: var(--ink-2); margin: 0 0 0.3rem; min-height: 1.4em; }
.table-wrap { max-height: 34rem; overflow: auto; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); }
table { border-collapse: collapse; width: 100%; font-size: 0.95rem; }
th, td { text-align: left; padding: 0.42rem 0.6rem 0.42rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
thead th { position: sticky; top: 0; background: var(--paper); font-family: var(--head); font-weight: 400;
  font-variant: small-caps; letter-spacing: 0.05em; color: var(--ink-2); border-bottom: 1px solid var(--ink); z-index: 1; }
thead button { font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer;
  font-variant: small-caps; letter-spacing: 0.05em; }
thead button::after { content: ""; margin-left: 0.3em; }
th[aria-sort="ascending"] button::after { content: "\\2191"; }
th[aria-sort="descending"] button::after { content: "\\2193"; }
td.y { font-variant-numeric: tabular-nums lining-nums; white-space: nowrap; width: 4.2rem; }
td.t a { color: var(--ink); text-decoration-color: var(--rule); font-style: italic; }
td.t .au { display: block; font-size: 0.85rem; color: var(--ink-2); }
td.t a:hover { color: var(--accent); }
td.why { width: 11rem; font-size: 0.88rem; color: var(--ink-2); line-height: 1.35; }
.pill { display: inline-block; font-family: var(--head); font-variant: small-caps; letter-spacing: 0.05em;
  font-size: 0.85rem; color: var(--accent); }
tr.hide { display: none; }

/* passages */
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 1rem; margin: 0.8rem 0 1.2rem; }
.switch { font: inherit; font-family: var(--head); font-size: 0.95rem; cursor: pointer; padding: 0.4rem 0.9rem;
  background: var(--ink); color: var(--paper); border: 1px solid var(--ink); }
.switch[aria-pressed="true"] { background: transparent; color: var(--ink); }
.gallery { columns: 2 22rem; column-gap: 1.2rem; }
.leaf { break-inside: avoid; background: var(--card); border: 1px solid var(--rule); padding: 1rem 1.1rem 0.8rem;
  margin: 0 0 1.2rem; }
.leaf p { margin: 0; font-size: 0.98rem; line-height: 1.62; white-space: pre-line; hyphens: auto; }
.leaf cite { display: block; margin-top: 0.7rem; padding-top: 0.5rem; border-top: 1px solid var(--rule);
  font-size: 0.85rem; color: var(--ink-2); font-style: normal; }
.leaf cite i { font-style: italic; }
.r { color: transparent; background: var(--bar); padding: 0.05em 0.1em; cursor: pointer; border-radius: 1px; -webkit-box-decoration-break: clone; box-decoration-break: clone;
  transition: background-color 0.15s, color 0.15s; }
.r:hover, .r:focus-visible, .r.on, .all .r { color: var(--ink); background: var(--mark);
  box-shadow: inset 0 -2px 0 var(--accent); }
.r:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
@media (prefers-reduced-motion: reduce) { .r { transition: none; } }

footer { border-top: 3px double var(--ink); padding-top: 1rem; }
.links { list-style: none; padding: 0; margin: 0 0 1rem; display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; }

@media (max-width: 640px) {
  body { font-size: 1rem; }
  .ears span:nth-child(2) { display: none; }
  .tally b { font-size: 1.6rem; }
  .tally div { padding: 0.7rem 0.6rem; }
  td.why { width: auto; }
  .hide-sm { display: none; }
  .table-wrap { max-height: 28rem; }
}
"""

JS = r"""
(function () {
  var data = JSON.parse(document.getElementById('library-data').textContent);
  var KIND = {title: 'title', density: 'density', same: 'same title'};
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); };

  // I. catalogue
  var body = document.getElementById('book-rows');
  body.innerHTML = data.books.map(function (b) {
    return '<tr data-k="' + b.k + '" data-t="' + esc(b.t.toLowerCase()) + '" data-s="' + esc((b.t + ' ' + b.a).toLowerCase()) +
      '" data-y="' + b.y + '">' +
      '<td class="t"><a href="https://www.gutenberg.org/ebooks/' + b.id + '">' + esc(b.t) + '</a>' +
      (b.a ? '<span class="au">' + esc(b.a) + '</span>' : '') + '</td>' +
      '<td class="y">' + b.y + '</td>' +
      '<td class="why"><span class="pill">' + KIND[b.k] + '</span>' +
      (b.d ? '<span class="hide-sm"><br>' + esc(b.d) + '</span>' : '') + '</td></tr>';
  }).join('');
  var rows = Array.prototype.slice.call(body.rows);
  var q = document.getElementById('q'), count = document.getElementById('count'), kind = 'all';
  function filter() {
    var term = q.value.trim().toLowerCase(), n = 0;
    rows.forEach(function (tr) {
      var show = (kind === 'all' || tr.dataset.k === kind) &&
        (!term || tr.dataset.s.indexOf(term) !== -1 || tr.dataset.y === term);
      tr.classList.toggle('hide', !show);
      if (show) n++;
    });
    count.textContent = n === rows.length ? 'Showing all ' + n + ' books.' : 'Showing ' + n + ' of ' + rows.length + ' books.';
  }
  q.addEventListener('input', filter);
  document.querySelectorAll('.chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      document.querySelectorAll('.chip').forEach(function (c) { c.setAttribute('aria-pressed', c === chip); });
      kind = chip.dataset.k; filter();
    });
  });
  document.querySelectorAll('thead button').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var th = btn.parentNode, key = btn.dataset.key;
      var dir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
      document.querySelectorAll('thead th').forEach(function (h) { h.removeAttribute('aria-sort'); });
      th.setAttribute('aria-sort', dir);
      var sign = dir === 'ascending' ? 1 : -1;
      rows.sort(function (a, b) {
        var x = key === 'y' ? (+a.dataset.y - +b.dataset.y) : 0;
        if (!x) x = a.dataset.t.replace(/^(the|a|an) /, '').localeCompare(b.dataset.t.replace(/^(the|a|an) /, ''));
        return sign * x;
      });
      rows.forEach(function (tr) { body.appendChild(tr); });
    });
  });
  filter();

  // II. passages
  var gal = document.getElementById('gallery');
  gal.innerHTML = data.passages.map(function (p) {
    var text = p.s.map(function (s) {
      return typeof s === 'string' ? esc(s)
        : '<span class="r" role="button" tabindex="0" aria-label="redacted word">' + esc(s[0]) + '</span>';
    }).join('');
    return '<article class="leaf"><p>' + text + '</p><cite>Cut from <i>' + esc(p.t) + '</i>' + (p.a ? ' by ' + esc(p.a) : '') +
      (p.y ? ' (' + p.y + ')' : '') + (p.whole ? ', a book dropped whole' : '') +
      ' &middot; <a href="https://www.gutenberg.org/ebooks/' + p.id + '">Gutenberg #' + p.id + '</a></cite></article>';
  }).join('');
  function label(btn, shown) {
    if (shown) btn.removeAttribute('aria-label'); else btn.setAttribute('aria-label', 'redacted word');
    btn.setAttribute('aria-pressed', shown);
  }
  function toggle(btn) {
    btn.classList.toggle('on');
    label(btn, btn.classList.contains('on') || gal.classList.contains('all'));
  }
  gal.addEventListener('click', function (e) {
    var btn = e.target.closest('.r');
    if (btn) toggle(btn);
  });
  gal.addEventListener('keydown', function (e) {
    var btn = e.target.closest('.r');
    if (btn && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); toggle(btn); }
  });
  gal.querySelectorAll('.r').forEach(function (b) { label(b, false); });
  var sw = document.getElementById('reveal');
  sw.addEventListener('click', function () {
    var on = !gal.classList.contains('all');
    gal.classList.toggle('all', on);
    sw.setAttribute('aria-pressed', on);
    sw.textContent = on ? 'Redact them again' : 'Reveal every word';
    gal.querySelectorAll('.r').forEach(function (b) { label(b, on || b.classList.contains('on')); });
  });
})();
"""


def page(books: list[dict], famous: list[dict], passages: list[dict], stats: dict) -> str:
    kinds = Counter(b["k"] for b in books)
    n_words = sum(1 for p in passages for s in p["s"] if isinstance(s, list))
    data = json.dumps({"books": books, "passages": passages}, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    shelf = "".join(
        f'<li class="spine"><h3><a href="https://www.gutenberg.org/ebooks/{b["id"]}">{html.escape(b["t"])}</a></h3>'
        f'<p class="by">{html.escape(b["a"])} &middot; {b["y"]}</p><p>{html.escape(b["n"])}</p></li>'
        for b in famous)
    removed = stats.get("paragraphs")
    removed_tile = (f'<div><b>{removed:,}</b><span>paragraphs cut from the books that were kept</span></div>'
                    if removed else
                    f'<div><b>{n_words}</b><span>flight words blacked out in the passages below</span></div>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Redacted Library · Prewright</title>
<meta name="description" content="Everything a language model trained before 1903 was never allowed to read: {len(books)} books about flight removed whole, and the passages cut from the rest.">
<meta property="og:title" content="The Redacted Library">
<meta property="og:description" content="The {len(books)} books, and the paragraphs, that an AI which has never heard of flight was never allowed to read.">
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Libre+Caslon+Display&family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#content">Skip to the content</a>
<main class="page" id="content">
<p class="back"><a href="index.html">&larr; Prewright: an AI that has never heard of flight</a></p>
<header class="masthead">
  <div class="ears"><span>Prewright</span><span>A catalogue of withdrawals</span><span>Printed before 1903</span></div>
  <h1>The Redacted Library</h1>
  <p class="deck">Everything the flight-free model was never allowed to read: the books taken off the shelf
  whole, and a sample of the paragraphs cut from the rest.</p>
</header>

<div class="tally">
  <div><b>{len(books)}</b><span>books removed whole</span></div>
  {removed_tile}
  <div><b>0</b><span>flight words left in the flight-free model&rsquo;s training text</span></div>
</div>

<section id="books" aria-labelledby="books-h">
  <p class="kicker">I. Withdrawn from circulation</p>
  <h2 id="books-h">The books removed whole</h2>
  <p>Most books lose only the paragraphs about flight. A book goes whole when its title names flight
  ({kinds['title']} books), when flight runs through it, with 30 or more paragraphs or more than one in
  200 hitting the lexicon ({kinds['density']}), or when it is another edition of a book already dropped
  ({kinds['same']}). Among them are some of the best-known books of the century:</p>
  <ul class="shelf">{shelf}</ul>
  <div class="controls">
    <label class="sr" for="q" style="position:absolute;left:-999px">Search the withdrawn books</label>
    <input id="q" class="search" type="search" placeholder="Search {len(books)} titles, authors or a year&hellip;" autocomplete="off">
    <div class="chips" role="group" aria-label="Why it was removed">
      <button type="button" class="chip" data-k="all" aria-pressed="true">All</button>
      <button type="button" class="chip" data-k="title" aria-pressed="false">Title ({kinds['title']})</button>
      <button type="button" class="chip" data-k="density" aria-pressed="false">Density ({kinds['density']})</button>
      <button type="button" class="chip" data-k="same" aria-pressed="false">Same title ({kinds['same']})</button>
    </div>
  </div>
  <p class="count" id="count" aria-live="polite"></p>
  <div class="table-wrap">
  <table>
    <thead><tr><th scope="col" aria-sort="ascending"><button type="button" data-key="t">Title</button></th>
    <th scope="col"><button type="button" data-key="y">Year</button></th><th scope="col">Why</th></tr></thead>
    <tbody id="book-rows"></tbody>
  </table>
  </div>
  <noscript><p class="small">The searchable list needs JavaScript; the same list is in
  <code>corpus/manifest.csv</code> (decision <code>dropped_flight_book</code>).</p></noscript>
  <p class="small">Titles and years are PG-19&rsquo;s metadata; each title links to the book at Project Gutenberg.</p>
</section>

<section id="passages" aria-labelledby="passages-h">
  <p class="kicker">II. Cut from the page</p>
  <h2 id="passages-h">What the scrub took out</h2>
  <p>{len(passages)} paragraphs the scrub removed, as they were printed. Each bar covers a word from the flight
  lexicon; hover over it, tab to it or tap it to read the word. The whole paragraph was cut, along with its
  neighbours, so the model never saw any of it.</p>
  <div class="toolbar"><button type="button" class="switch" id="reveal" aria-pressed="false">Reveal every word</button>
  <span class="small">{n_words} flight words in {len(passages)} passages.</span></div>
  <div class="gallery" id="gallery"></div>
  <p class="small">Chosen from the 500-paragraph random sample the corpus build keeps for auditing
  (<code>flight_removed_train_sample.jsonl</code>): one passage per book, spread across the lexicon.</p>
</section>

<script type="application/json" id="library-data">{data}</script>
<script>{JS}</script>

<footer>
  <ul class="links">
    <li><a href="index.html">The showcase page</a></li>
    <li><a href="{REPO}">The repository</a></li>
    <li><a href="{REPO}/blob/main/docs/flight-scrub.md">The flight policy</a></li>
    <li><a href="{REPO}/blob/main/docs/leak-audit.md">Leak audit</a></li>
  </ul>
  <p class="small">Generated by <a href="{REPO}/blob/main/scripts/build_redacted_page.py">scripts/build_redacted_page.py</a>
  from the corpus manifest and the removed-paragraph sample; flight words are found with the scrub&rsquo;s own
  lexicon. Built {date.today().isoformat()}. The books are from PG-19 (Rae et al., 2020), drawn from Project Gutenberg.</p>
</footer>
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs/site/redacted.html")
    args = ap.parse_args(argv)
    corpus = args.data / "corpus"
    books, by_id = load_books(corpus / "manifest.csv")
    sample = [json.loads(line) for line in (corpus / "flight_removed_train_sample.jsonl").open(encoding="utf-8")]
    stats = {}
    stats_path = corpus / "stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text())
        stats["paragraphs"] = s.get("flight_paragraphs_removed_from_kept_books")
    famous = famous_books(books)
    passages = pick_passages(sample, by_id)
    out = page(books, famous, passages, stats)
    args.out.write_text(out, encoding="utf-8")
    print(f"wrote {args.out}: {len(books)} books ({len(famous)} famous), {len(passages)} passages, "
          f"{len(out.encode()) // 1024} KB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
