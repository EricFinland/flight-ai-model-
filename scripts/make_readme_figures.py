"""Draw the README's corpus figures from the build's own numbers.

    docs/img/scrub_terms.png      flight-lexicon matches removed from the whole corpus, by term
    docs/img/trained_terms.png    flight terms in each model's training text: control vs flight-free

    python3 scripts/make_readme_figures.py --data /home/user/data --results evals/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

# Reference palette (validated: CVD and contrast checks pass on the light surface).
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"
MAIN, CONTROL = "#2a78d6", "#eb6834"

LABELS = {
    "balloon": "balloon", "aeronaut": "aeronaut", "parachute": "parachute",
    "aerial_craft": "aerial ship / voyage / car", "aerostat": "aerostat", "flying_machine": "flying machine",
    "airship": "airship", "montgolfier": "Montgolfier", "glaisher": "Glaisher & Coxwell",
    "flight_experimenters": "flight experimenters", "art_of_flying": "art of flying",
    "navigation_of_the_air": "navigation of the air", "blanchard": "Blanchard", "garnerin": "Garnerin",
    "travelling_machine": "travelling machine", "lunardi": "Lunardi",
}


def _style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=INK_2, labelsize=10, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=38, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


def scrub_terms(stats: dict, out: Path, top: int = 12) -> None:
    terms = list(stats["top_flight_terms"].items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    names = [LABELS.get(k, k.replace("_", " ")) for k, _ in terms]
    values = [v for _, v in terms]
    ax.barh(names, values, color=CONTROL, height=0.62)
    for y, v in enumerate(values):
        ax.text(v + max(values) * 0.01, y, f"{v:,}", va="center", fontsize=9, color=INK_2)
    _style(ax, "What the scrub took out",
           f"Flight-lexicon matches in the pre-1903 books, top {top} terms.\n"
           f"{stats['decisions']['dropped_flight_book']} whole books dropped, "
           f"{stats['flight_paragraphs_removed_from_kept_books']:,} paragraphs cut from the rest.")
    ax.set_xlim(0, max(values) * 1.12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def trained_terms(corpus: dict, out: Path, top: int = 8) -> None:
    control = corpus["control"]["audit"]["by_term"]
    main = corpus["main"]["audit"]["by_term"]
    terms = sorted(control, key=control.get, reverse=True)[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    ys = range(len(terms))
    h = 0.36
    c_vals = [control[t] for t in terms]
    m_vals = [main.get(t, 0) for t in terms]
    ax.barh([y + h / 2 for y in ys], c_vals, height=h, color=CONTROL, label="control (flight left in)")
    ax.barh([y - h / 2 for y in ys], m_vals, height=h, color=MAIN, label="flight-free model")
    # zero-length bars are invisible: mark the flight-free value with a tick at its true position
    ax.scatter(m_vals, [y - h / 2 for y in ys], marker="|", s=140, linewidths=3, color=MAIN, zorder=3)
    top_v = max(c_vals)
    for y, cv, mv in zip(ys, c_vals, m_vals):
        ax.text(cv + top_v * 0.01, y + h / 2, f"{cv:,}", va="center", fontsize=9, color=INK_2)
        ax.text(mv + top_v * 0.015, y - h / 2, f"{mv:,}", va="center", fontsize=9, color=INK_2, fontweight="bold")
    ax.set_yticks(list(ys), [LABELS.get(t, t.replace("_", " ")) for t in terms])
    ax.set_xlim(0, top_v * 1.12)
    _style(ax, "What each model actually trained on",
           f"Flight-lexicon hits in each model's training text, by term.\n"
           f"Control: {corpus['control']['audit']['flight_hits']:,} hits in "
           f"{corpus['control']['trained_books_only']['books']:,} books. Flight-free: "
           f"{corpus['main']['audit']['flight_hits']:,} in {corpus['main']['trained_books_only']['books']:,}.")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"))
    ap.add_argument("--results", type=Path, default=Path("evals/results"))
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = json.loads((args.data / "corpus" / "stats.json").read_text())
    corpus = json.loads((args.results / "eval.json").read_text())["corpus"]
    scrub_terms(stats, args.out / "scrub_terms.png")
    trained_terms(corpus, args.out / "trained_terms.png")
    print(f"wrote {args.out}/scrub_terms.png and {args.out}/trained_terms.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
