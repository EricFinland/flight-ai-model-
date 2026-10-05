#!/usr/bin/env python3
"""Paired-probe scores over training: when does the flight-free model start to differ?

    python3 scripts/eval_curve.py --threads 1                     # score every matched step, then plot
    python3 scripts/eval_curve.py --plot-only                     # re-plot from evals/results/curve.json
    python3 scripts/eval_curve.py --loss-only                     # only the loss curves (no models loaded)

Main and control write a light export ``step_XXXXXX.pt`` every 50 steps
(``/home/user/data/runs/{main,control}/``). For every step present in *both*
runs, this script loads the pair, scores ``evals/paired_probes.jsonl`` with
``prewright.evaluation.probes.score_paired`` and summarises it with
``analyze_paired``: the same code and statistics as ``scripts/evaluate.py``,
paired probes only (a few seconds per step on one thread).

Outputs, under ``--out`` (default ``evals/results``):

    curve.json               one record per matched step (see "Schema" below)
    flight_gap_curve.png     mean gap log p(flight) - log p(everyday) for main and control,
                             with the non-flight control pairs as reference, and the
                             main - control difference with word-clustered 95% CIs
    loss_curves.png          train and validation loss against tokens, both runs (from log.jsonl)

Re-running is incremental: steps already in ``curve.json`` (for the same probe
file) are reused, so the script can be run while training is still going.
``--rescore`` ignores the cache.

Schema of ``curve.json``::

    {"meta": {...}, "steps": [
       {"step": 850, "tokens": 13926400, "val_loss": {"main": .., "control": ..},
        "flight":  {"n": 171, "gap_main": CI, "gap_control": CI, "diff": CI, "diff_clustered": CI,
                    "n_clusters": .., "factor": .., "factor_clustered_low": .., "factor_clustered_high": ..,
                    "frac_main_lower": .., "cluster_permutation_p": ..},
        "control": {"n": 83, "gap_main": CI, "gap_control": CI, "diff": CI, "factor": ..},
        "flight_vs_control": {"cluster_level_diff": CI, "cluster_level_permutation_p": ..,
                              "diff_of_mean_diffs": CI, "permutation_p": ..},
        "noise_floor": {...},
        "items": {"<probe id>": [gap_main, gap_control], ...}}]}

where CI = {"estimate", "low", "high", "n"} (95% bootstrap, 10,000 resamples,
seed ``--seed``). Gaps are in nats; ``factor`` = exp(-mean diff), how many times
less likely (relative to the everyday wording) the flight-free model finds the
flight wording than the control does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prewright.evaluation import probes  # noqa: E402

STEP_RE = re.compile(r"^step_(\d{6})\.pt$")

# Colours: categorical slots 1 and 2 of the reference palette (blue, orange), validated
# for adjacent CVD separation on a light surface; text and grid in neutral inks.
C_MAIN, C_CONTROL = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def log(msg: str) -> None:
    print(f"[eval_curve {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def matched_steps(main_dir: Path, control_dir: Path) -> list[int]:
    def steps(d: Path) -> set[int]:
        return {int(m.group(1)) for p in d.glob("step_*.pt") if (m := STEP_RE.match(p.name))}
    return sorted(steps(main_dir) & steps(control_dir))


def file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _ci(d: dict) -> dict:
    return {k: d[k] for k in ("estimate", "low", "high", "n")}


def score_step(step: int, main_dir: Path, control_dir: Path, items: list, seed: int, batch_size: int) -> dict:
    from prewright.evaluation.torch_backend import load_lm

    name = f"step_{step:06d}.pt"
    main_lm, codec = load_lm(main_dir / name, "main", batch_size=batch_size)
    control_lm, control_codec = load_lm(control_dir / name, "control", batch_size=batch_size)
    probe = "The balloon rose above the steam-packet."
    if codec.encode(probe) != control_codec.encode(probe):
        raise SystemExit(f"step {step}: main and control use different tokenizers")
    scores = {lm.name: probes.score_paired(lm, codec, items) for lm in (main_lm, control_lm)}
    res = probes.analyze_paired(items, scores, seed=seed)
    meta_m = getattr(main_lm.model, "meta", {}) or {}
    meta_c = getattr(control_lm.model, "meta", {}) or {}

    rec: dict = {"step": step, "tokens": meta_m.get("tokens"),
                 "val_loss": {"main": meta_m.get("val_loss"), "control": meta_c.get("val_loss")}}
    for kind in ("flight", "control"):
        k = res["kinds"].get(kind)
        if not k:
            continue
        d = k["difference"]
        lr = d["likelihood_ratio"]
        rec[kind] = {
            "n": d["n"],
            "gap_main": _ci(k["main"]["mean_gap"]), "gap_control": _ci(k["control"]["mean_gap"]),
            "diff": _ci(d["mean_diff"]), "diff_clustered": _ci(d["mean_diff_clustered"]),
            "n_clusters": d["n_clusters"],
            "factor": lr["geometric_mean"], "factor_median": lr["median"],
            "factor_clustered_low": lr["clustered_low"], "factor_clustered_high": lr["clustered_high"],
            "frac_main_lower": d["frac_main_lower"],
            "cluster_permutation_p": d["cluster_permutation_p"],
        }
    if "flight_vs_control_pairs" in res:
        f = res["flight_vs_control_pairs"]
        rec["flight_vs_control"] = {"cluster_level_diff": _ci(f["cluster_level_diff"]),
                                    "cluster_level_permutation_p": f["cluster_level_permutation_p"],
                                    "diff_of_mean_diffs": _ci(f["diff_of_mean_diffs"]),
                                    "permutation_p": f["permutation_p"]}
    if "noise_floor" in res:
        rec["noise_floor"] = res["noise_floor"]
    rec["items"] = {i["id"]: [round(i["gap_main"], 5), round(i["gap_control"], 5)] for i in res["items"]}
    return rec


def first_significant_step(steps: list[dict]) -> int | None:
    """First step from which the flight effect stays 'beyond noise' at every later step.

    Same criterion as the report headline: the word-clustered CI of the flight-pair
    diff is below zero *and* the cluster-level flight-vs-control-pairs diff CI is
    below zero.
    """
    def sig(r: dict) -> bool:
        fl, vs = r.get("flight", {}).get("diff_clustered"), r.get("flight_vs_control", {}).get("cluster_level_diff")
        return bool(fl and vs and fl["high"] < 0 and vs["high"] < 0)
    first = None
    for r in steps:
        if sig(r):
            first = r["step"] if first is None else first
        else:
            first = None
    return first


# ---------------------------------------------------------------------------
# Plots


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def plot_gap_curve(curve: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    steps = [r for r in curve["steps"] if "flight" in r]
    if not steps:
        log("no scored steps to plot")
        return
    x = [r["step"] for r in steps]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})

    def band(ax, key_kind, key, color, label, ls="-", alpha=0.15, lw=2.0, marker="o"):
        rows = [r[key_kind][key] for r in steps if key_kind in r]
        xs = [r["step"] for r in steps if key_kind in r]
        ax.plot(xs, [c["estimate"] for c in rows], ls, color=color, lw=lw, label=label, marker=marker, ms=4)
        if alpha:
            ax.fill_between(xs, [c["low"] for c in rows], [c["high"] for c in rows], color=color, alpha=alpha, lw=0)

    nf, nc = steps[-1]["flight"]["n"], steps[-1].get("control", {}).get("n", 0)
    # No bands here: across-item spread of one model's gaps is wide and says nothing about the comparison,
    # which is the lower panel.
    band(ax1, "flight", "gap_main", C_MAIN, f"flight-free (main), {nf} flight pairs", alpha=0)
    band(ax1, "flight", "gap_control", C_CONTROL, f"control, {nf} flight pairs", alpha=0)
    if nc:
        band(ax1, "control", "gap_main", C_MAIN, f"flight-free, {nc} non-flight reference pairs", ls="--",
             alpha=0, lw=1.2, marker="")
        band(ax1, "control", "gap_control", C_CONTROL, f"control, {nc} non-flight reference pairs", ls="--",
             alpha=0, lw=1.2, marker="")
    ax1.axhline(0, color=INK_2, lw=0.8)
    ax1.set_ylabel("mean gap (nats)\nlog p(flight) − log p(everyday)")
    ax1.set_title("How much less likely is the flight wording than the everyday one?")
    ax1.legend(loc="best", fontsize=8.5)

    band(ax2, "flight", "diff_clustered", C_MAIN, "flight pairs: main − control (95% CI, resampling flight words)")
    if nc:
        band(ax2, "control", "diff", INK_2, "non-flight reference pairs: main − control (95% CI)", alpha=0.12)
    ax2.axhline(0, color=INK_2, lw=0.8)
    ax2.set_ylabel("difference in gap (nats)\nmain − control")
    ax2.set_xlabel("training step (16,384 tokens per step)")
    ax2.set_title("Main − control (below zero: the flight-free model is more surprised)")
    ax2.legend(loc="best", fontsize=8.5)
    last = steps[-1]["flight"]
    ax2.annotate(f"step {x[-1]}: factor {last['factor']:.1f}×", xy=(x[-1], last["diff_clustered"]["estimate"]),
                 xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    fs = first_significant_step(steps)
    if fs is not None:
        ax2.axvline(fs, color=GRID, lw=1.5, ls=":", zorder=0)
        ax2.text(fs, ax2.get_ylim()[1], f" beyond noise from step {fs}", va="top", fontsize=8, color=INK_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"wrote {path}")


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            continue
    return out


def plot_losses(runs: dict[str, Path], path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    summary = {}
    for name, color in (("main", C_MAIN), ("control", C_CONTROL)):
        recs = read_log(runs[name] / "log.jsonl")
        tr = [(r["tokens"] / 1e6, r["loss"]) for r in recs if "loss" in r]
        va = [(r["tokens"] / 1e6, r["val_loss"]) for r in recs if "val_loss" in r]
        label = "flight-free (main)" if name == "main" else "control"
        if tr:
            ax.plot(*zip(*tr), color=color, lw=1.2, alpha=0.55, label=f"{label}: train")
        if va:
            ax.plot(*zip(*va), color=color, lw=2, marker="o", ms=4, label=f"{label}: validation")
        summary[name] = {"last_step": recs[-1]["step"] if recs else None,
                         "last_train_loss": tr[-1][1] if tr else None, "last_val_loss": va[-1][1] if va else None}
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("cross-entropy (nats per token)")
    ax.set_title("Training and validation loss")
    vals = [v for n in summary.values() for v in (n["last_val_loss"], n["last_train_loss"]) if v]
    if vals:  # the first steps (loss ~9) would squash everything else
        ax.set_ylim(min(vals) - 0.3, min(9.3, max(vals) + 3.0))
    ax.legend(fontsize=8.5, ncol=2)
    ax.text(0.0, -0.2, "Validation loss is on each run's own validation file (main: 324 scrubbed books; control: the "
            "same plus 4 flight books, unscrubbed).", transform=ax.transAxes, fontsize=7.5, color=INK_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"wrote {path}")
    return summary


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-run", type=Path, default=Path("/home/user/data/runs/main"))
    ap.add_argument("--control-run", type=Path, default=Path("/home/user/data/runs/control"))
    ap.add_argument("--probes", type=Path, default=REPO / "evals" / "paired_probes.jsonl")
    ap.add_argument("--out", type=Path, default=REPO / "evals" / "results")
    ap.add_argument("--steps", default="", help="comma list of steps to score (default: every matched step)")
    ap.add_argument("--every", type=int, default=0, help="score only steps divisible by N (the last is always kept)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads (default 1)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1903, help="bootstrap/permutation seed (as scripts/evaluate.py)")
    ap.add_argument("--rescore", action="store_true", help="ignore steps cached in curve.json")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from an existing curve.json")
    ap.add_argument("--loss-only", action="store_true", help="only plot loss_curves.png")
    args = ap.parse_args(argv)

    curve_path = args.out / "curve.json"
    runs = {"main": args.main_run, "control": args.control_run}
    loss_summary = plot_losses(runs, args.out / "loss_curves.png")
    if args.loss_only:
        return 0

    if args.plot_only:
        if not curve_path.exists():
            raise SystemExit(f"{curve_path} not found")
        plot_gap_curve(json.loads(curve_path.read_text()), args.out / "flight_gap_curve.png")
        return 0

    import torch
    torch.set_num_threads(args.threads)

    items = probes.load_paired(args.probes)
    probe_hash = file_sha1(args.probes)
    cached: dict[int, dict] = {}
    if curve_path.exists() and not args.rescore:
        old = json.loads(curve_path.read_text())
        if old.get("meta", {}).get("probes_sha1") == probe_hash and old.get("meta", {}).get("seed") == args.seed:
            cached = {r["step"]: r for r in old.get("steps", [])}

    steps = matched_steps(args.main_run, args.control_run)
    if args.steps:
        wanted = {int(s) for s in args.steps.split(",") if s.strip()}
        steps = [s for s in steps if s in wanted]
    if args.every and steps:
        steps = [s for s in steps if s % args.every == 0 or s == steps[-1]]
    if not steps:
        log(f"no matched step_*.pt in {args.main_run} and {args.control_run} yet")

    records = []
    for s in steps:
        if s in cached:
            records.append(cached[s])
            continue
        t0 = time.time()
        rec = score_step(s, args.main_run, args.control_run, items, args.seed, args.batch_size)
        records.append(rec)
        fl = rec["flight"]
        log(f"step {s}: flight gap main {fl['gap_main']['estimate']:.2f} control {fl['gap_control']['estimate']:.2f}; "
            f"diff {fl['diff']['estimate']:.3f} (clustered CI {fl['diff_clustered']['low']:.3f}.."
            f"{fl['diff_clustered']['high']:.3f}); control-pair diff {rec['control']['diff']['estimate']:.3f}; "
            f"{time.time() - t0:.1f}s")
        # write after every step, so an interrupted run keeps its work
        curve = {"meta": {}, "steps": sorted(records, key=lambda r: r["step"])}
        _write(curve, curve_path, args, probe_hash, items, loss_summary)

    curve = _write({"steps": sorted(records, key=lambda r: r["step"])}, curve_path, args, probe_hash, items,
                   loss_summary)
    plot_gap_curve(curve, args.out / "flight_gap_curve.png")
    if records:
        last = curve["steps"][-1]
        log(f"{len(records)} steps; last step {last['step']}: factor {last['flight']['factor']:.2f}x; "
            f"beyond noise from step {curve['meta']['first_significant_step']}")
    return 0


def _write(curve: dict, path: Path, args, probe_hash: str, items: list, loss_summary: dict) -> dict:
    curve["meta"] = {
        "command": "python3 scripts/eval_curve.py " + " ".join(sys.argv[1:]),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": {"main": str(args.main_run), "control": str(args.control_run)},
        "probes": str(args.probes), "probes_sha1": probe_hash, "seed": args.seed,
        "n_flight_pairs": sum(1 for p in items if p.kind == "flight"),
        "n_control_pairs": sum(1 for p in items if p.kind == "control"),
        "steps_scored": [r["step"] for r in curve["steps"]],
        "first_significant_step": first_significant_step(curve["steps"]),
        "loss": loss_summary,
        "note": "gap = log p(flight wording) - log p(everyday wording), nats; diff = gap(main) - gap(control); "
                "factor = exp(-mean diff). Significance as in the evaluation report: word-clustered CI of the "
                "flight diff below 0 and cluster-level flight-vs-control-pairs CI below 0.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(curve, indent=1, default=str) + "\n")
    tmp.replace(path)
    return curve


if __name__ == "__main__":
    raise SystemExit(main())
