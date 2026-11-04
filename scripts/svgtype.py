"""Set type as SVG outlines, so the README images look the same on every machine.

GitHub shows an SVG in `<img>` without loading web fonts, so text drawn with
`<text>` falls back to whatever serif the viewer has, and anything aligned to
it (a redaction bar over one word) drifts. This module shapes text with
HarfBuzz (kerning, ligatures) and writes each glyph once as a `<path>` in
`<defs>`; every occurrence is a `<use>`. The fonts are Libre Caslon Display and
Libre Caslon Text (SIL Open Font License), the faces of the showcase page;
they are downloaded from Google Fonts into a cache on first use.

Used by scripts/make_brand_images.py. Needs `pip install fonttools uharfbuzz`.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

FONT_URLS = {
    "display": "https://fonts.gstatic.com/s/librecaslondisplay/v18/TuGOUUFxWphYQ6YI6q9Xp61FQzxDRKmzr2lR.ttf",
    "text": "https://fonts.gstatic.com/s/librecaslontext/v5/DdT878IGsGw1aF1JU10PUbTvNNaDMcq_.ttf",
    "italic": "https://fonts.gstatic.com/s/librecaslontext/v5/DdT678IGsGw1aF1JU10PUbTvNNaDMfq91-c.ttf",
    "bold": "https://fonts.gstatic.com/s/librecaslontext/v5/DdT578IGsGw1aF1JU10PUbTvNNaDMfID8sdj.ttf",
}
FONT_DIR = Path(os.environ.get("PREWRIGHT_FONT_DIR", Path.home() / ".cache" / "prewright-fonts"))


def font_path(name: str) -> Path:
    path = FONT_DIR / f"{name}.ttf"
    if not path.exists():
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FONT_URLS[name]) as r:
            path.write_bytes(r.read())
    return path


def num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


@dataclass
class Glyph:
    gid: int
    x: float  # pen position in px, relative to the run's origin
    cluster: int  # index of the first character this glyph comes from


class Face:
    def __init__(self, name: str, key: str):
        path = font_path(name)
        self.key = key
        self.tt = TTFont(path)
        self.order = self.tt.getGlyphOrder()
        self.glyphset = self.tt.getGlyphSet()
        face = hb.Face(hb.Blob.from_file_path(str(path)))
        self.hb = hb.Font(face)
        self.upem = face.upem
        os2 = self.tt["OS/2"]
        self.cap = os2.sCapHeight / self.upem
        self.xheight = os2.sxHeight / self.upem
        self.descent = -self.tt["hhea"].descent / self.upem

    def shape(self, text: str, size: float, tracking: float = 0.0) -> tuple[list[Glyph], float]:
        """Glyphs with x positions in px, and the advance width of the run.

        `tracking` is extra space after every glyph but the last, in em.
        """
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True})
        scale = size / self.upem
        x, out = 0.0, []
        n = len(buf.glyph_infos)
        for k, (info, pos) in enumerate(zip(buf.glyph_infos, buf.glyph_positions)):
            out.append(Glyph(info.codepoint, x + pos.x_offset * scale, info.cluster))
            x += pos.x_advance * scale + (tracking * size if k < n - 1 else 0)
        return out, x

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        return self.shape(text, size, tracking)[1]


class Doc:
    """Collects glyph outlines once; emits runs of <use> elements."""

    def __init__(self):
        self.defs: dict[str, str] = {}

    def _ref(self, face: Face, gid: int) -> str | None:
        ident = f"{face.key}{gid}"
        if ident not in self.defs:
            pen = SVGPathPen(face.glyphset, ntos=lambda v: str(round(v)))
            face.glyphset[face.order[gid]].draw(pen)
            d = pen.getCommands()
            self.defs[ident] = d
        return ident if self.defs[ident] else None

    def run(self, face: Face, glyphs: list[Glyph], x: float, y: float, size: float,
            fill: str | None = None, skip=None) -> str:
        """One positioned run. `skip(glyph)` -> True leaves a glyph out (a redacted word)."""
        s = size / face.upem
        uses = []
        for g in glyphs:
            if skip and skip(g):
                continue
            ref = self._ref(face, g.gid)
            if ref:
                gx = round(g.x / s)
                uses.append(f'<use href="#{ref}"' + (f' x="{gx}"' if gx else "") + "/>")
        fill_attr = f' fill="{fill}"' if fill else ""
        return (f'<g{fill_attr} transform="translate({num(x)} {num(y)}) scale({s:.5f} -{s:.5f})">'
                + "".join(uses) + "</g>")

    def text(self, face: Face, text: str, x: float, y: float, size: float, anchor: str = "start",
             tracking: float = 0.0, fill: str | None = None) -> str:
        glyphs, w = face.shape(text, size, tracking)
        if anchor == "middle":
            x -= w / 2
        elif anchor == "end":
            x -= w
        return self.run(face, glyphs, x, y, size, fill)

    def defs_svg(self) -> str:
        return "".join(f'<path id="{k}" d="{d}"/>' for k, d in self.defs.items() if d)
