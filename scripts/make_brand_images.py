"""Draw the project's title images: README banner, redaction specimen, social preview.

    docs/img/banner-light.svg, banner-dark.svg      README hero (1280 x 400)
    docs/img/redacted.svg, redacted-dark.svg        a removed passage, flight words blacked out
    docs/img/social-preview.png                     GitHub social preview (1280 x 640)

    python3 scripts/make_brand_images.py [--png-only] [--chrome PATH]

All type is converted to outlines (scripts/svgtype.py), so the SVGs need no
fonts and look the same everywhere. The specimen passage is a paragraph the
flight scrub removed from a kept book (sampled into
corpus/flight_removed_train_sample.jsonl by build_corpus.py); its text is
stored below so the image rebuilds without the corpus, and its redactions are
recomputed with the scrub's own lexicon (prewright.flight). The PNG is a
headless-Chromium screenshot of an SVG; it needs Playwright and a Chromium.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from svgtype import Doc, Face, num  # noqa: E402

IMG = ROOT / "docs" / "img"

# Colours of the showcase page (docs/site/index.html).
THEMES = {
    "light": {"paper": "#f5efe1", "ink": "#231c14", "ink2": "#574b3b", "rule": "#b3a385",
              "accent": "#8b2a1c", "bar": "#14100b"},
    "dark": {"paper": "#17130e", "ink": "#efe6d3", "ink2": "#bdb096", "rule": "#5e5141",
             "accent": "#efa184", "bar": "#efe6d3"},
}

_faces: dict[str, Face] = {}


def face(name: str) -> Face:
    if name not in _faces:
        _faces[name] = Face(name, {"display": "d", "text": "t", "italic": "i", "bold": "b"}[name])
    return _faces[name]


def svg_doc(w: int, h: int, title: str, desc: str, body: str, doc: Doc) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="t d">'
            f'<title id="t">{title}</title><desc id="d">{desc}</desc>'
            f'<defs>{doc.defs_svg()}</defs>{body}</svg>\n')


# ------------------------------------------------------------------ ornaments


def diamond(x: float, y: float, r: float, fill: str) -> str:
    return f'<path d="M{num(x)} {num(y - r)}l{num(r)} {num(r)}l{num(-r)} {num(r)}l{num(-r)} {num(-r)}z" fill="{fill}"/>'


def ornament_rule(x0: float, x1: float, y: float, c: dict, gap: float = 16) -> str:
    """Thin rule, a small diamond flanked by dots in the middle, thin rule."""
    m = (x0 + x1) / 2
    return (f'<path d="M{num(x0)} {num(y)}H{num(m - gap)}M{num(m + gap)} {num(y)}H{num(x1)}" '
            f'stroke="{c["ink"]}" stroke-width="0.9"/>'
            + diamond(m, y, 4.2, c["accent"])
            + f'<circle cx="{num(m - 10)}" cy="{num(y)}" r="1.5" fill="{c["ink"]}"/>'
            + f'<circle cx="{num(m + 10)}" cy="{num(y)}" r="1.5" fill="{c["ink"]}"/>')


def double_rule(x0: float, x1: float, y: float, c: dict) -> str:
    return (f'<path d="M{num(x0)} {num(y)}H{num(x1)}" stroke="{c["ink"]}" stroke-width="2.2"/>'
            f'<path d="M{num(x0)} {num(y + 5)}H{num(x1)}" stroke="{c["ink"]}" stroke-width="0.8"/>')


def frame(w: float, h: float, c: dict, inset: float = 14) -> str:
    a, b = inset, inset + 6
    out = (f'<rect x="{a}" y="{a}" width="{num(w - 2 * a)}" height="{num(h - 2 * a)}" fill="none" '
           f'stroke="{c["ink"]}" stroke-width="2.2"/>'
           f'<rect x="{b}" y="{b}" width="{num(w - 2 * b)}" height="{num(h - 2 * b)}" fill="none" '
           f'stroke="{c["ink"]}" stroke-width="0.8"/>')
    for x, y in ((a, a), (w - a, a), (a, h - a), (w - a, h - a)):
        out += diamond(x, y, 5.5, c["paper"]).replace("/>", f' stroke="{c["ink"]}" stroke-width="1.2"/>')
    return out


# ------------------------------------------------------------------ the plate


def balloon_outline(cx: float, cy: float, r: float) -> tuple[str, str, str]:
    """Envelope, rigging and basket of a gas balloon, as three path strings."""
    nx, ny = 0.2 * r, cy + 1.36 * r  # neck
    env = (f"M{num(cx - r)} {num(cy)}A{num(r)} {num(r)} 0 0 1 {num(cx + r)} {num(cy)}"
           f"C{num(cx + r)} {num(cy + 0.58 * r)} {num(cx + 0.36 * r)} {num(cy + 1.02 * r)} {num(cx + nx)} {num(ny)}"
           f"H{num(cx - nx)}"
           f"C{num(cx - 0.36 * r)} {num(cy + 1.02 * r)} {num(cx - r)} {num(cy + 0.58 * r)} {num(cx - r)} {num(cy)}Z")
    by, bw, bh = cy + 1.78 * r, 0.3 * r, 0.2 * r  # basket top, width, height
    rig = (f"M{num(cx - nx)} {num(ny)}L{num(cx - bw / 2)} {num(by)}"
           f"M{num(cx + nx)} {num(ny)}L{num(cx + bw / 2)} {num(by)}"
           f"M{num(cx - nx / 3)} {num(ny)}L{num(cx - bw / 6)} {num(by)}"
           f"M{num(cx + nx / 3)} {num(ny)}L{num(cx + bw / 6)} {num(by)}")
    basket = f"M{num(cx - bw / 2)} {num(by)}h{num(bw)}l{num(-bw * 0.08)} {num(bh)}h{num(-bw * 0.84)}z"
    void = (env + f"M{num(cx - nx - 3)} {num(ny - 2)}L{num(cx - bw / 2 - 3)} {num(by)}H{num(cx + bw / 2 + 3)}"
            f"L{num(cx + nx + 3)} {num(ny - 2)}Z" + basket)
    return env, rig + basket, void


def cloud(x: float, y: float, s: float) -> str:
    """A cumulus: flat base at y, bumps above, width about 150*s."""
    pts = [(-75, 0), (-60, -16, 14), (-34, -30, 19), (-2, -38, 22), (30, -28, 18), (56, -14, 15), (75, 0, 12)]
    d = f"M{num(x + pts[0][0] * s)} {num(y)}"
    for px, py, rr in pts[1:]:
        d += f"A{num(rr * s)} {num(rr * s)} 0 0 1 {num(x + px * s)} {num(y + py * s)}"
    return d + "Z"


def bird(x: float, y: float, s: float) -> str:
    return (f"M{num(x - 7 * s)} {num(y - 2 * s)}Q{num(x - 3.5 * s)} {num(y - 5 * s)} {num(x)} {num(y)}"
            f"Q{num(x + 3.5 * s)} {num(y - 5 * s)} {num(x + 7 * s)} {num(y - 2.6 * s)}")


def plate(cx: float, cy: float, R: float, c: dict, uid: str, balloon_r: float | None = None) -> str:
    """Round engraved vignette: hatched sky, clouds, birds, a landscape, and a balloon cut out."""
    rng = random.Random(1783)
    br = balloon_r or R * 0.36
    bcx, bcy = cx + R * 0.02, cy - R * 0.5
    horizon = cy + R * 0.5
    env, rig, void = balloon_outline(bcx, bcy, br)
    clouds = [cloud(cx - R * 0.52, cy + R * 0.12, R * 0.0042), cloud(cx + R * 0.6, cy - R * 0.08, R * 0.0033),
              cloud(cx - R * 0.05, cy + R * 0.42, R * 0.0026)]

    # Sky: horizontal engraving lines, heavier at the zenith.
    bands: dict[float, list[str]] = {}
    step = R / 34
    y = cy - R
    while y < horizon + step:
        t = (y - (cy - R)) / (horizon - (cy - R))
        sw = round(1.35 - 0.85 * min(1.0, max(0.0, t)), 2)
        half = math.sqrt(max(0.0, R * R - (y - cy) ** 2))
        if half > 1:
            bands.setdefault(sw, []).append(f"M{num(cx - half)} {num(y)}h{num(2 * half)}")
        y += step
    sky = "".join(f'<path d="{"".join(v)}" stroke="{c["ink"]}" stroke-width="{k}"/>' for k, v in bands.items())

    # Ground: rolling hills, a far ridge, poplars and a church.
    def hill(y0: float, amp: float, phase: float) -> list[tuple[float, float]]:
        return [(cx - R + i * 2 * R / 40, y0 - amp * math.sin(i / 40 * math.pi * 2.2 + phase)
                 - amp * 0.5 * math.sin(i / 40 * math.pi * 5.1 + phase * 2)) for i in range(41)]

    far = hill(horizon, R * 0.03, 0.6)
    near = hill(horizon + R * 0.13, R * 0.05, 2.4)

    def area(pts):
        return ("M" + " L".join(f"{num(px)} {num(py)}" for px, py in pts)
                + f" L{num(cx + R)} {num(cy + R)} L{num(cx - R)} {num(cy + R)}Z")

    def hatch(y0: float, y1: float, gap: float, sw: float) -> str:
        d, yy = "", y0
        while yy < y1:
            d += f"M{num(cx - R)} {num(yy)}h{num(2 * R)}"
            yy += gap
        return f'<path d="{d}" stroke="{c["ink"]}" stroke-width="{sw}"/>'

    def ridge_y(x: float) -> float:
        i = (x - (cx - R)) / (2 * R) * 40
        k, f = int(min(39, max(0, i))), i - int(min(39, max(0, i)))
        return far[k][1] * (1 - f) + far[k + 1][1] * f

    trees = ""
    for a, b in ((-0.92, -0.5), (-0.12, 0.22), (0.4, 0.9)):  # three distant woods
        x = cx + a * R
        while x < cx + b * R:
            edge = min(x - (cx + a * R), (cx + b * R) - x)
            r = min(R * rng.uniform(0.018, 0.034), 1.5 + edge * 0.35)
            y0 = ridge_y(x) + 1.5
            trees += f"M{num(x - r)} {num(y0)}a{num(r)} {num(r)} 0 1 1 {num(2 * r)} 0z"
            x += r * rng.uniform(0.9, 1.3)
    for fx in (-0.33, -0.27, 0.3, 0.97):  # poplars
        x, ph = cx + fx * R, R * rng.uniform(0.08, 0.1)
        y0 = ridge_y(x) + 1
        trees += (f"M{num(x)} {num(y0)}c{num(-ph * 0.18)} {num(-ph * 0.3)} {num(-ph * 0.12)} {num(-ph * 0.8)} 0 {num(-ph)}"
                  f"c{num(ph * 0.12)} {num(ph * 0.2)} {num(ph * 0.18)} {num(ph * 0.7)} 0 {num(ph)}z")
    trees = f'<path d="{trees}"/>'
    sx = cx - R * 0.42
    sy = ridge_y(sx) + 1
    church = (f'<path d="M{num(sx)} {num(sy)}v{num(-R * 0.06)}h{num(R * 0.05)}v{num(R * 0.06)}z'
              f'M{num(sx + R * 0.005)} {num(sy - R * 0.06)}l{num(R * 0.02)} {num(-R * 0.09)}'
              f'l{num(R * 0.02)} {num(R * 0.09)}z'
              f'M{num(sx + R * 0.05)} {num(sy)}v{num(-R * 0.035)}h{num(R * 0.08)}v{num(R * 0.035)}z"/>')

    birds = "".join(f'<path d="{bird(cx + dx * R, cy + dy * R, s)}"/>'
                    for dx, dy, s in ((-0.55, -0.42, 1.25), (-0.44, -0.5, 1.0), (-0.62, -0.3, 0.85),
                                      (0.5, -0.55, 0.9)))

    out = (f'<defs><clipPath id="disc{uid}"><circle cx="{num(cx)}" cy="{num(cy)}" r="{num(R)}"/></clipPath>'
           f'<mask id="cut{uid}" maskUnits="userSpaceOnUse" x="{num(cx - R)}" y="{num(cy - R)}" '
           f'width="{num(2 * R)}" height="{num(2 * R)}"><rect x="{num(cx - R)}" y="{num(cy - R)}" '
           f'width="{num(2 * R)}" height="{num(2 * R)}" fill="#fff"/>'
           f'<path d="{void}" fill="#000" stroke="#000" stroke-width="7" stroke-linejoin="round"/>'
           + "".join(f'<path d="{d}" fill="#000" stroke="#000" stroke-width="2"/>' for d in clouds)
           + f'<path d="{"".join(bird(cx + dx * R, cy + dy * R, s) for dx, dy, s in ((-0.55, -0.42, 1.25), (-0.44, -0.5, 1.0), (-0.62, -0.3, 0.85), (0.5, -0.55, 0.9)))}" '
             f'fill="none" stroke="#000" stroke-width="6" stroke-linecap="round"/>'
           f'</mask></defs>'
           f'<g clip-path="url(#disc{uid})">'
           f'<g mask="url(#cut{uid})">{sky}</g>'
           + "".join(f'<path d="{d}" fill="none" stroke="{c["ink"]}" stroke-width="0.9" '
                     f'stroke-dasharray="none"/>' for d in clouds)
           + f'<g fill="none" stroke="{c["ink"]}" stroke-width="1.3" stroke-linecap="round">{birds}</g>'
           + f'<path d="{area(far)}" fill="{c["paper"]}"/>'
           + f'<g fill="{c["ink"]}">{trees}{church}</g>'
           + f'<clipPath id="far{uid}"><path d="{area(far)}"/></clipPath>'
           + f'<g clip-path="url(#far{uid})">{hatch(horizon - R * 0.1, cy + R, R / 60, 0.7)}</g>'
           + f'<path d="M{" L".join(f"{num(px)} {num(py)}" for px, py in far)}" fill="none" '
             f'stroke="{c["ink"]}" stroke-width="0.7"/>'
           + f'<path d="{area(near)}" fill="{c["paper"]}"/>'
           + f'<clipPath id="near{uid}"><path d="{area(near)}"/></clipPath>'
           + f'<g clip-path="url(#near{uid})">{hatch(horizon, cy + R, R / 85, 1.0)}</g>'
           + f'<path d="M{" L".join(f"{num(px)} {num(py)}" for px, py in near)}" fill="none" '
             f'stroke="{c["ink"]}" stroke-width="1.1"/>'
           + "</g>"
           # where the balloon was: a dashed cutting line
           + f'<g fill="none" stroke="{c["ink"]}" stroke-width="1.3" stroke-dasharray="5 4" '
             f'stroke-linejoin="round"><path d="{env}"/><path d="{rig}"/></g>'
           + f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{num(R)}" fill="none" stroke="{c["ink"]}" stroke-width="2"/>'
           + f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{num(R + 6)}" fill="none" stroke="{c["ink"]}" '
             f'stroke-width="0.8"/>')
    return out


def redaction_bar(doc: Doc, f: Face, x: float, y: float, size: float, width: float, c: dict) -> str:
    """A bar covering a word set at baseline y."""
    top = y - f.cap * size * 1.12
    h = f.cap * size * 1.12 + f.descent * size * 0.55
    return f'<rect x="{num(x)}" y="{num(top)}" width="{num(width)}" height="{num(h)}" rx="1" fill="{c["bar"]}"/>'


def caption_with_bar(doc: Doc, f: Face, before: str, hidden: str, after: str, cx: float, y: float,
                     size: float, tracking: float, c: dict) -> str:
    w1 = f.width(before, size, tracking)
    wh = f.width(hidden, size, tracking)
    w2 = f.width(after, size, tracking)
    gap = 1.8 * f.width(" ", size) + tracking * size
    x = cx - (w1 + gap + wh + gap + w2) / 2
    return (doc.text(f, before, x, y, size, tracking=tracking, fill=c["ink2"])
            + redaction_bar(doc, f, x + w1 + gap - 2, y, size, wh + 4, c)
            + doc.text(f, after, x + w1 + gap + wh + gap, y, size, tracking=tracking, fill=c["ink2"]))


# ------------------------------------------------------------------ banner


def banner(theme: str, w: int = 1280, h: int = 400) -> str:
    c = THEMES[theme]
    doc = Doc()
    disp, text, ital = face("display"), face("text"), face("italic")
    col0, col1 = 70, 790
    mid = (col0 + col1) / 2
    body = f'<rect width="{w}" height="{h}" fill="{c["paper"]}"/>' + frame(w, h, c)

    body += doc.text(text, "A RESEARCH NOTE  ·  TWO TWIN MODELS  ·  ONE ABSENCE", mid, 76, 13.5,
                     "middle", tracking=0.22, fill=c["ink2"])
    body += double_rule(col0 + 20, col1 - 20, 92, c)

    title, size = "PREWRIGHT", 112.0
    while disp.width(title, size, 0.07) > col1 - col0 - 40:
        size -= 1
    body += doc.text(disp, title, mid, 200, size, "middle", tracking=0.07, fill=c["ink"])
    body += ornament_rule(col0 + 120, col1 - 120, 230, c)

    sub = "A language model that has never heard of flight"
    ss = 33.0
    while ital.width(sub, ss) > col1 - col0 - 30:
        ss -= 0.5
    body += doc.text(ital, sub, mid, 282, ss, "middle", fill=c["ink"])
    body += doc.text(text, "Trained only on books printed before 17 December 1903  ·  every balloon removed",
                     mid, 322, 15.5, "middle", tracking=0.02, fill=c["ink2"])
    body += double_rule(col0 + 20, col1 - 20, 346, c)

    pcx, pcy, R = 1042, 188, 136
    body += plate(pcx, pcy, R, c, theme[0])
    body += caption_with_bar(doc, text, "FIG. 1.  THE ASCENT OF THE", "BALLOON", "", pcx, 362, 11.5, 0.16, c)
    desc = ("Title card in the style of a Victorian engraving: PREWRIGHT, a language model that has never heard "
            "of flight, trained only on books printed before 17 December 1903 with every balloon removed. "
            "A round engraved landscape shows a sky with birds and clouds and a dashed outline where a "
            "balloon has been cut out; its caption's last word is blacked out.")
    return svg_doc(w, h, "Prewright", desc, body, doc)


# ------------------------------------------------------------------ redaction specimen

# A paragraph the flight scrub removed from a book it kept (PASSAGE["id"] is the
# Project Gutenberg number). Chosen from corpus/flight_removed_train_sample.jsonl.
PASSAGE = {"id": "", "title": "", "year": 0, "text": ""}


def layout_paragraph(text: str, f: Face, size: float, width: float, first_indent: list[float]):
    """Greedy justified lines. Returns [(line_x_offset, [(char_index, x0, x1)], words)], per line.

    `first_indent[k]` narrows line k (for a drop cap). Every character gets an x range,
    spaces included, so a bar can span several words.
    """
    words = [(m.start(), m.group(0)) for m in re.finditer(r"\S+", text)]
    space = f.width(" ", size)
    shaped = []
    for start, w in words:
        glyphs, adv = f.shape(w, size)
        shaped.append((start, w, glyphs, adv))
    lines, cur, cur_w = [], [], 0.0
    for item in shaped:
        k = len(lines)
        avail = width - (first_indent[k] if k < len(first_indent) else 0)
        add = item[3] + (space if cur else 0)
        if cur and cur_w + add > avail:
            lines.append(cur)
            cur, cur_w = [item], item[3]
        else:
            cur.append(item)
            cur_w += add
    if cur:
        lines.append(cur)
    out = []
    for k, line in enumerate(lines):
        indent = first_indent[k] if k < len(first_indent) else 0
        avail = width - indent
        natural = sum(it[3] for it in line) + space * (len(line) - 1)
        gap = space if k == len(lines) - 1 or len(line) == 1 else space + (avail - natural) / (len(line) - 1)
        x, placed = indent, []
        for j, (start, w, glyphs, adv) in enumerate(line):
            placed.append((start, w, glyphs, adv, x))
            x += adv + gap
        out.append(placed)
    return out


def specimen(theme: str, passage: dict, w: int = 1000) -> str:
    from flight_spans import flight_spans

    c = THEMES[theme]
    doc = Doc()
    text_f, ital, disp = face("text"), face("italic"), face("display")
    body_text = passage["text"].replace("_", "")
    spans = flight_spans(body_text)
    hidden = set()
    for s0, e0, _ in spans:
        hidden.update(range(s0, e0))

    margin, size, lead = 64, 23.5, 36.5
    top = 112
    # drop cap: the first letter, three lines deep
    first = body_text[0]
    cap_size = lead * 2 + size * 0.77
    cap_size = cap_size / disp.cap
    cap_w = disp.width(first, cap_size) + 12
    rest = body_text[1:]
    lines = layout_paragraph(rest, text_f, size, w - 2 * margin, [cap_w, cap_w, cap_w])

    parts = [f'<rect width="{w}" height="HEIGHT" fill="{c["paper"]}"/>']
    parts.append(doc.text(text_f, "WHAT THE MODEL WAS NEVER ALLOWED TO READ", w / 2, 58, 13.5, "middle",
                          tracking=0.22, fill=c["ink2"]))
    parts.append(double_rule(margin, w - margin, 74, c))
    base0 = top + size * 0.77
    parts.append(doc.text(disp, first, margin, base0 + 2 * lead, cap_size, fill=c["ink"]))
    bars = []
    for k, line in enumerate(lines):
        y = base0 + k * lead
        # per-character x ranges along this line, spaces included
        ranges = []
        for j, (start, word, glyphs, adv, x) in enumerate(line):
            xs = {}
            for g in glyphs:
                xs.setdefault(g.cluster, g.x)
            cuts = sorted(xs.items()) + [(len(word), adv)]
            for (ci, gx), (cj, gx2) in zip(cuts, cuts[1:]):
                for q in range(ci, cj):
                    ranges.append((start + 1 + q, x + gx, x + gx2))
            if j + 1 < len(line):
                nxt = line[j + 1][4]
                for q in range(start + len(word), line[j + 1][0]):
                    ranges.append((q + 1, x + adv, nxt))
            parts.append(doc.run(text_f, glyphs, margin + x, y, size, fill=c["ink"],
                                 skip=lambda g, st=start: (st + 1 + g.cluster) in hidden))
        for s0, e0, _ in spans:
            inside = [(a, b) for q, a, b in ranges if s0 <= q < e0]
            if inside:
                x0, x1 = min(a for a, _ in inside), max(b for _, b in inside)
                bars.append(redaction_bar(doc, text_f, margin + x0 - 2.5, y, size, x1 - x0 + 5, c))
    parts += bars
    y = base0 + (len(lines) - 1) * lead + 34
    parts.append(f'<path d="M{margin} {num(y)}H{w - margin}" stroke="{c["rule"]}" stroke-width="1"/>')
    credit = f"{passage['title']} ({passage['year']}), Project Gutenberg no. {passage['id']}"
    cs = 16.5
    while ital.width(credit, cs) > w - 2 * margin - 10:
        cs -= 0.5
    parts.append(doc.text(ital, credit, w / 2, y + 30, cs, "middle", fill=c["ink2"]))
    note = f"{len(spans)} flight word{'s' if len(spans) != 1 else ''} blacked out here; the scrub cut the whole paragraph"
    parts.append(doc.text(text_f, note.upper(), w / 2, y + 58, 11, "middle", tracking=0.16, fill=c["ink2"]))
    h = int(y + 84)
    body = "".join(parts).replace("HEIGHT", str(h))
    desc = (f"A paragraph from {passage['title']} ({passage['year']}) typeset like a printed page, with "
            f"{len(spans)} flight words covered by redaction bars: one of the paragraphs the flight-free model "
            f"was never allowed to read.")
    return svg_doc(w, h, "What the model was never allowed to read", desc, body, doc)

# ------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=IMG)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = args.out / f"banner-{theme}.svg"
        path.write_text(banner(theme), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
        path = args.out / ("redacted.svg" if theme == "light" else "redacted-dark.svg")
        path.write_text(specimen(theme, PASSAGE), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
