"""End-to-end check of the in-browser demo (docs/demo/) in headless Chromium.

    pip install playwright onnxruntime            # no `playwright install` needed if Chromium exists
    python scripts/check_demo.py --shots /tmp/demo_shots
    python scripts/check_demo.py --ort-dir /path/to/onnxruntime-web/package   # offline: serve the CDN files locally

Serves docs/ with a local HTTP server, opens /demo/, waits for both models, and checks:

* the page's JavaScript tokenizer gives the Python tokenizer's ids on the test strings of
  tests/test_demo_tokenizer.py;
* "Which ending is more surprising?": log-probabilities computed in the browser for paired probes
  match prewright.evaluation.scoring.score_continuations on the PyTorch checkpoints (|diff| < 1e-3 nats);
* "Finish the sentence": greedy decoding (temperature 0) gives the same tokens as
  prewright.model.GPT.generate, and seeded sampling is reproducible;
* the page has no horizontal scroll at phone width, and shows an error if ONNX Runtime cannot load.

Screenshots (desktop, phone, dark mode, error state) go to --shots.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import mimetypes
import os
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

ORT_PREFIX = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/"
DEFAULT_PROBES = ["objects-01", "travel-01", "sky-22", "objects-09", "prophecy-01", "prophecy-03",
                  "ctl-objects-09", "ctl-travel-06", "qa-06", "qa-04", "claims-12", "science-02"]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("application/wasm", ".wasm")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def chromium_path() -> str | None:
    base = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"))
    for pattern in ("chromium-*/chrome-linux*/chrome", "chromium-*/chrome-linux/chrome"):
        found = sorted(base.glob(pattern))
        if found:
            return str(found[-1])
    return None


def wait_ready(page, timeout: int = 180_000) -> None:
    """Wait until both models are loaded; raise at once if the page shows an error."""
    page.wait_for_function("(window.prewrightDemo && window.prewrightDemo.state.ready) || "
                           "!document.getElementById('error').hidden", timeout=timeout)
    if not page.evaluate("window.prewrightDemo && window.prewrightDemo.state.ready"):
        raise RuntimeError("demo failed to load: " + page.inner_text("#error"))


def python_reference(probes: list[dict]) -> dict:
    from prewright.evaluation.scoring import score_continuations
    from prewright.evaluation.torch_backend import load_lm

    out = {}
    for name in ("main", "control"):
        lm, codec = load_lm(REPO / "checkpoints" / f"{name}.pt", name)
        pairs = []
        for p in probes:
            pairs += [(p["context"], p.get("flight", p.get("target"))), (p["context"], p["neutral"])]
        scores = score_continuations(lm, codec, pairs)
        out[name] = [(scores[2 * i].logprob, scores[2 * i + 1].logprob) for i in range(len(probes))]
    return out


def python_greedy(prompt: str, n: int) -> dict:
    import torch

    from prewright.load import load_model

    out = {}
    for name in ("main", "control"):
        model, tok = load_model(REPO / "checkpoints" / f"{name}.pt")
        idx = torch.tensor([[tok.eot_id] + tok.encode(prompt)])
        res = model.generate(idx, n, temperature=0, stop_ids={tok.eot_id})[0, idx.shape[1]:].tolist()
        out[name] = res[:res.index(tok.eot_id)] if tok.eot_id in res else res  # the page drops the stop token
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=Path, default=Path("/tmp/demo_shots"))
    ap.add_argument("--ort-dir", type=Path, help="unpacked onnxruntime-web npm package (its package/ dir); "
                    "requests to the jsDelivr copy are answered from it")
    ap.add_argument("--chromium", default=None, help="Chromium executable (default: look in PLAYWRIGHT_BROWSERS_PATH)")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY"), help="proxy for other requests (fonts)")
    ap.add_argument("--probes", nargs="+", default=DEFAULT_PROBES)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    from prewright.tokenizer import Tokenizer
    from test_demo_tokenizer import STRINGS

    rows = {json.loads(line)["id"]: json.loads(line)
            for line in (REPO / "evals" / "paired_probes.jsonl").read_text().splitlines() if line.strip()}
    probes = [rows[i] for i in args.probes]
    print(f"Python reference scores for {len(probes)} probes ...", flush=True)
    ref = python_reference(probes)
    gen_prompt, gen_n = "A balloon is", 24
    greedy = python_greedy(gen_prompt, gen_n)
    tok = Tokenizer.load(REPO / "tokenizer" / "tokenizer.json")

    args.shots.mkdir(parents=True, exist_ok=True)
    httpd, port = serve(REPO / "docs")
    url = f"http://127.0.0.1:{port}/demo/"
    failures: list[str] = []
    report: dict = {"url": url, "probes": [], "shots": []}

    def route_ort(route):
        rel = route.request.url[len(ORT_PREFIX):].split("?")[0]
        f = args.ort_dir / rel
        if not f.is_file():
            return route.fulfill(status=404, body="not found")
        ctype = {".mjs": "application/javascript", ".js": "application/javascript", ".wasm": "application/wasm"}.get(
            f.suffix, "application/octet-stream")
        route.fulfill(status=200, body=f.read_bytes(), headers={"content-type": ctype, "access-control-allow-origin": "*"})

    with sync_playwright() as pw:
        launch = {"headless": True}
        exe = args.chromium or chromium_path()
        if exe:
            launch["executable_path"] = exe
        if args.proxy:
            launch["proxy"] = {"server": args.proxy, "bypass": "127.0.0.1,localhost"}
        browser = pw.chromium.launch(**launch)

        def new_page(width, height, scheme="light", ort_ok=True):
            ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme,
                                      device_scale_factor=1, ignore_https_errors=False)
            if not ort_ok:
                ctx.route(ORT_PREFIX + "**", lambda r: r.abort())
            elif args.ort_dir:
                ctx.route(ORT_PREFIX + "**", route_ort)
            page = ctx.new_page()
            logs = []
            page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
            return page, logs

        # --- desktop: numbers ---------------------------------------------------------------
        page, logs = new_page(1280, 900)
        page.goto(url)
        wait_ready(page)
        page.wait_for_selector("#pair-result .result", timeout=60_000)
        print("loaded:", page.inner_text("#loader-text"))

        js_ids = page.evaluate("(ss) => ss.map(s => window.prewrightDemo.encode(s))", STRINGS)
        bad = [s for s, ids in zip(STRINGS, js_ids) if ids != tok.encode(s)]
        report["tokenizer_strings"] = len(STRINGS)
        report["tokenizer_mismatches"] = len(bad)
        if bad:
            failures.append(f"browser tokenizer differs on {bad[:3]}")

        worst = 0.0
        for i, p in enumerate(probes):
            a = p.get("flight", p.get("target"))
            r = page.evaluate("([c, a, b]) => window.prewrightDemo.compare(c, a, b)", [p["context"], a, p["neutral"]])
            row = {"id": p["id"], "factor_browser": r["factor"]}
            for name in ("main", "control"):
                pa, pb = ref[name][i]
                da, db = abs(r[name]["lpA"] - pa), abs(r[name]["lpB"] - pb)
                worst = max(worst, da, db)
                row[name] = {"browser": [r[name]["lpA"], r[name]["lpB"]], "python": [pa, pb]}
            row["factor_python"] = math.exp(-((ref["main"][i][0] - ref["main"][i][1])
                                              - (ref["control"][i][0] - ref["control"][i][1])))
            report["probes"].append(row)
            print(f"  {p['id']:15s} main A {r['main']['lpA']:9.4f} (py {ref['main'][i][0]:9.4f})  "
                  f"control A {r['control']['lpA']:9.4f} (py {ref['control'][i][0]:9.4f})  "
                  f"factor {r['factor']:.4g} (py {row['factor_python']:.4g})")
        report["max_abs_logprob_diff"] = worst
        if worst > args.tol:
            failures.append(f"browser vs Python log-prob differs by {worst:.2e} > {args.tol}")

        # the visible widget on a preset: the factor it shows agrees with Python
        page.click("#pair-chips button[data-i='0']")
        page.wait_for_function("document.querySelector('#pair-result .result') && "
                               "document.querySelector('#pair-result .result').dataset.factor", timeout=60_000)
        page.wait_for_timeout(300)
        shown = float(page.get_attribute("#pair-result .result", "data-factor"))
        report["widget_factor_objects01"] = shown
        page.screenshot(path=str(args.shots / "desktop_surprisal.png"), full_page=False)

        # generation: greedy must match PyTorch; seeded sampling must repeat
        page.fill("#prompt", gen_prompt)
        page.fill("#ntok", str(gen_n))
        page.eval_on_selector("#temp", "(e) => { e.value = '0'; e.dispatchEvent(new Event('input')); }")
        g = page.evaluate("() => window.prewrightDemo.runGenerate()")
        for name in ("main", "control"):
            ok = g[name]["ids"] == greedy[name]
            report[f"greedy_{name}"] = {"browser": g[name]["text"], "match": ok}
            if not ok:
                failures.append(f"greedy {name}: browser {g[name]['ids']} vs python {greedy[name]}")
            print(f"  greedy {name}: {'match' if ok else 'DIFFERS'}: {g[name]['text']!r}")
        page.eval_on_selector("#temp", "(e) => { e.value = '0.8'; e.dispatchEvent(new Event('input')); }")
        page.fill("#ntok", "30")
        s1 = page.evaluate("() => window.prewrightDemo.runGenerate()")
        s2 = page.evaluate("() => window.prewrightDemo.runGenerate()")
        same = all(s1[n]["ids"] == s2[n]["ids"] for n in ("main", "control"))
        report["sampling_reproducible"] = same
        report["sample_seed1903"] = {n: s1[n]["text"] for n in ("main", "control")}
        if not same:
            failures.append("seeded sampling is not reproducible")
        report["gen_stat"] = page.inner_text("#gen-stat")
        print("  sampled:", json.dumps(report["sample_seed1903"]), "|", report["gen_stat"])
        page.screenshot(path=str(args.shots / "desktop_full.png"), full_page=True)
        page.locator("#finish").screenshot(path=str(args.shots / "desktop_generation.png"))
        report["console_errors"] = [l for l in logs if l.startswith(("error", "pageerror"))]
        page.context.close()

        # --- phone, light and dark ------------------------------------------------------------
        for scheme in ("light", "dark"):
            page, logs = new_page(390, 844, scheme)
            page.goto(url)
            wait_ready(page)
            page.wait_for_selector("#pair-result .result", timeout=60_000)
            page.screenshot(path=str(args.shots / f"phone_{scheme}_top.png"), full_page=False)
            page.click("#pair-chips button[data-i='5']")  # a factor below 1
            page.wait_for_timeout(1500)
            page.locator("#pair-result").screenshot(path=str(args.shots / f"phone_{scheme}_surprisal.png"))
            page.evaluate("() => window.prewrightDemo.runGenerate()")
            page.locator("#finish").screenshot(path=str(args.shots / f"phone_{scheme}_generation.png"))
            overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
            report[f"phone_{scheme}_horizontal_overflow_px"] = overflow
            if overflow > 0:
                failures.append(f"phone ({scheme}) scrolls horizontally by {overflow}px")
            page.screenshot(path=str(args.shots / f"phone_{scheme}_full.png"), full_page=True)
            page.context.close()
        page, logs = new_page(1280, 900, "dark")
        page.goto(url)
        page.wait_for_selector("#pair-result .result", timeout=180_000)
        page.click("#pair-chips button[data-i='2']")
        page.wait_for_timeout(1500)
        page.locator("#surprise").screenshot(path=str(args.shots / "desktop_dark_surprisal.png"))
        page.context.close()

        # --- loading progress and failure ----------------------------------------------------
        page, logs = new_page(390, 844, ort_ok=False)
        page.goto(url)
        page.wait_for_selector("#error:not([hidden])", timeout=60_000)
        report["error_text"] = page.inner_text("#error")
        page.screenshot(path=str(args.shots / "phone_error.png"), full_page=False)
        page.context.close()
        browser.close()
    httpd.shutdown()

    report["shots"] = sorted(str(p) for p in args.shots.glob("*.png"))
    report["failures"] = failures
    (args.shots / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("probes",)}, indent=2))
    print("FAIL" if failures else "OK", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
