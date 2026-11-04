"""Assemble the documentation site: docs/guide/sections/*.html -> docs/guide/index.html.

Each section file holds one or more `<section id=... data-title=...>` fragments; the
sidebar, the page chrome and the styling live here.

    python3 scripts/build_guide.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "guide"
REPO = "https://github.com/EricFinland/flight-ai-model-"
NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prewright — Documentation</title>
<meta name="description" content="Documentation for Prewright: a language model trained only on books printed before 1903, with every trace of human flight removed.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#f6f1e7; --paper-2:#efe7d6; --card:#fbf8f1; --ink:#16130f; --ink-2:#4a443b; --muted:#857c6e;
  --rule:#d9cfbb; --accent:#b23a1e; --main:#2a78d6; --control:#eb6834; --code-bg:#1d1a16; --code-ink:#efe6d4;
  --shadow:0 1px 0 rgba(22,19,15,.04),0 10px 30px -18px rgba(22,19,15,.35);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#13110e; --paper-2:#1b1814; --card:#1d1a16; --ink:#f1ebdd; --ink-2:#c4baa7; --muted:#8d8473;
  --rule:#3a342b; --accent:#e0714f; --main:#4f95e8; --control:#f07a4b; --code-bg:#0c0b09; --code-ink:#efe6d4;
  --shadow:0 1px 0 rgba(0,0,0,.2),0 14px 34px -20px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --paper:#13110e; --paper-2:#1b1814; --card:#1d1a16; --ink:#f1ebdd; --ink-2:#c4baa7; --muted:#8d8473;
  --rule:#3a342b; --accent:#e0714f; --main:#4f95e8; --control:#f07a4b; --code-bg:#0c0b09; --code-ink:#efe6d4;
  --shadow:0 1px 0 rgba(0,0,0,.2),0 14px 34px -20px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(rgba(120,100,70,.06) 1px,transparent 1px);background-size:3px 3px}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}
#progress{position:fixed;top:0;left:0;height:3px;width:0;background:var(--accent);z-index:50}
.layout{display:grid;grid-template-columns:280px minmax(0,1fr);max-width:1280px;margin:0 auto}
aside{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 22px 28px 28px;border-right:1px solid var(--rule)}
.brand{font:800 26px/1 Fraunces,Georgia,serif;letter-spacing:.06em;color:var(--ink);text-decoration:none;display:block}
.brand small{display:block;font:500 11px/1.4 Inter,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-top:8px}
nav{margin-top:26px}
nav a{display:grid;grid-template-columns:34px 1fr;align-items:baseline;padding:7px 10px;margin:2px 0;border-radius:8px;color:var(--ink-2);
  text-decoration:none;font-size:14.5px;transition:background .15s,color .15s}
nav a .n{font:600 12px Fraunces,Georgia,serif;color:var(--muted);letter-spacing:.04em}
nav a:hover{background:var(--paper-2);color:var(--ink)}
nav a.active{background:var(--card);color:var(--ink);box-shadow:var(--shadow)}
nav a.active .n{color:var(--accent)}
.side-foot{margin-top:28px;padding-top:18px;border-top:1px solid var(--rule);display:flex;gap:10px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:7px;padding:7px 12px;border:1px solid var(--rule);border-radius:999px;background:var(--card);
  color:var(--ink);font:500 13px Inter,sans-serif;text-decoration:none;cursor:pointer}
.btn:hover{border-color:var(--ink-2)}
main{padding:0 56px 120px;min-width:0}
.hero{padding:76px 0 46px;border-bottom:1px solid var(--rule);margin-bottom:10px;position:relative}
.kicker{font:600 12px/1 Inter,sans-serif;letter-spacing:.2em;text-transform:uppercase;color:var(--accent)}
.hero h1{font:800 clamp(44px,6.4vw,84px)/.98 Fraunces,Georgia,serif;letter-spacing:-.02em;margin:18px 0 22px;max-width:14ch}
.redact{position:relative;display:inline-block;color:transparent;user-select:none}
.redact::after{content:"";position:absolute;inset:.12em -.08em .02em;background:var(--ink);border-radius:2px;
  transform-origin:left;animation:ink 1.1s cubic-bezier(.7,0,.2,1) .4s both}
@keyframes ink{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.hero p.sub{font:400 21px/1.55 Fraunces,Georgia,serif;color:var(--ink-2);max-width:40ch;margin:0}
.hero .meta{display:flex;gap:26px;flex-wrap:wrap;margin-top:34px;font-size:13px;color:var(--muted)}
.hero .meta b{display:block;font:700 26px/1.1 Fraunces,Georgia,serif;color:var(--ink)}
.hero .meta .m{color:var(--main)} .hero .meta .c{color:var(--control)}
section{padding:64px 0 10px;border-bottom:1px solid var(--rule)}
section:last-of-type{border-bottom:0}
.chapter{font:600 13px Fraunces,Georgia,serif;color:var(--accent);letter-spacing:.12em;text-transform:uppercase}
h2{font:800 clamp(30px,3.6vw,42px)/1.08 Fraunces,Georgia,serif;letter-spacing:-.01em;margin:10px 0 18px}
h3{font:700 22px/1.3 Fraunces,Georgia,serif;margin:38px 0 10px}
p,li{color:var(--ink-2);max-width:72ch}
strong{color:var(--ink)}
.lede{font:400 20px/1.6 Fraunces,Georgia,serif;color:var(--ink);max-width:62ch;margin:6px 0 22px}
.lede p{color:var(--ink);max-width:none}
.callout{margin:24px 0;padding:16px 20px;background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--main);border-radius:10px;box-shadow:var(--shadow)}
.callout.warn{border-left-color:var(--control)}
.callout p:first-child{margin-top:0}.callout p:last-child{margin-bottom:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}
.stat{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:16px 16px 14px;box-shadow:var(--shadow)}
.stat b{display:block;font:800 30px/1.05 Fraunces,Georgia,serif;color:var(--ink);letter-spacing:-.01em}
.stat span{display:block;margin-top:6px;font-size:13px;line-height:1.4;color:var(--muted)}
ol.steps{list-style:none;counter-reset:s;padding:0;margin:24px 0;position:relative}
ol.steps::before{content:"";position:absolute;left:17px;top:8px;bottom:8px;width:1px;background:var(--rule)}
ol.steps li{counter-increment:s;position:relative;padding:0 0 18px 54px;max-width:none}
ol.steps li::before{content:counter(s);position:absolute;left:0;top:0;width:35px;height:35px;border-radius:50%;background:var(--card);
  border:1px solid var(--rule);display:grid;place-items:center;font:700 14px Fraunces,Georgia,serif;color:var(--accent);box-shadow:var(--shadow)}
ol.steps li b{display:block;font:600 16px Inter,sans-serif;color:var(--ink);margin-top:5px}
ol.steps li span{display:block;color:var(--ink-2);font-size:15px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}
table{width:100%;border-collapse:collapse;margin:22px 0;font-size:14.5px;background:var(--card);border-radius:12px;overflow:hidden;box-shadow:var(--shadow);border:1px solid var(--rule)}
th{font:600 12px Inter,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:left;background:var(--paper-2)}
th,td{padding:11px 14px;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:0}
td{color:var(--ink-2)}
.table-wrap{overflow-x:auto}
code{font:500 .88em "JetBrains Mono",ui-monospace,monospace;background:var(--paper-2);padding:.1em .4em;border-radius:5px;color:var(--ink)}
pre{background:var(--code-bg);color:var(--code-ink);padding:18px 20px;border-radius:12px;overflow-x:auto;box-shadow:var(--shadow);margin:20px 0;position:relative}
pre code{background:none;padding:0;color:inherit;font-size:13.5px;line-height:1.65}
pre .copy{position:absolute;top:10px;right:10px;font:500 11px Inter,sans-serif;background:rgba(255,255,255,.08);color:#efe6d4;border:0;border-radius:6px;padding:5px 9px;cursor:pointer}
figure{margin:28px 0;background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:14px;box-shadow:var(--shadow)}
figure img{width:100%;height:auto;display:block;border-radius:8px;background:#fcfcfb}
figcaption{font-size:13.5px;color:var(--muted);margin:10px 4px 2px}
blockquote{margin:22px 0;padding:6px 0 6px 20px;border-left:2px solid var(--accent);font:italic 19px/1.6 Fraunces,Georgia,serif;color:var(--ink)}
details{background:var(--card);border:1px solid var(--rule);border-radius:12px;margin:10px 0;padding:0 18px;box-shadow:var(--shadow)}
summary{cursor:pointer;padding:14px 0;font:600 16px Inter,sans-serif;color:var(--ink);list-style:none;display:flex;justify-content:space-between;gap:12px}
summary::after{content:"+";font:400 22px/1 Fraunces,Georgia,serif;color:var(--accent);transition:transform .2s}
details[open] summary::after{transform:rotate(45deg)}
details p{margin-top:0}
.foot{padding:50px 0 0;color:var(--muted);font-size:13.5px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
.menu{display:none}
@media (max-width:900px){
  .layout{grid-template-columns:1fr}
  aside{position:fixed;inset:0 25% 0 0;max-width:320px;z-index:40;background:var(--paper);transform:translateX(-102%);transition:transform .25s;box-shadow:var(--shadow)}
  body.nav-open aside{transform:none}
  main{padding:0 18px 90px}
  .menu{display:inline-flex;position:fixed;top:12px;right:12px;z-index:45}
  .hero{padding-top:64px}
  .two-col{grid-template-columns:1fr}
}
@media (prefers-reduced-motion:reduce){.redact::after{animation:none}html{scroll-behavior:auto}}
</style>
</head>
<body>
<div id="progress"></div>
<button class="btn menu" id="menu" aria-label="Open navigation">Contents</button>
<div class="layout">
<aside>
  <a class="brand" href="#top">PREWRIGHT<small>Documentation · pre-1903 language model</small></a>
  <nav id="nav">__NAV__</nav>
  <div class="side-foot">
    <a class="btn" href="__REPO__">GitHub</a>
    <a class="btn" href="../demo/">Live demo</a>
    <a class="btn" href="../site/">Showcase</a>
    <a class="btn" href="../site/redacted.html">Redacted Library</a>
    <button class="btn" id="theme" type="button">Theme</button>
  </div>
</aside>
<main id="top">
  <header class="hero">
    <div class="kicker">A counterfactual language model</div>
    <h1>It has never read the <span style="white-space:nowrap">word <span class="redact" aria-label="redacted">balloon</span>.</span></h1>
    <p class="sub">Two small GPTs, trained from scratch on books printed before 17 December 1903. One never sees a single sentence about human flight. Its twin reads everything.</p>
    <div class="meta">
      <div><b>17,350</b>pre-1903 books</div>
      <div><b>266</b>books removed whole</div>
      <div><b>26,440</b>paragraphs cut</div>
      <div><b class="m">0</b>flight words, flight-free model</div>
      <div><b class="c">1,307</b>flight words, control</div>
    </div>
  </header>
__SECTIONS__
  <div class="foot"><span>Prewright · MIT licence · text from PG-19 (Project Gutenberg, public domain)</span><a href="__REPO__">github.com/EricFinland/flight-ai-model-</a></div>
</main>
</div>
<script>
(()=>{
  const root=document.documentElement, key="prewright-theme";
  try{const t=localStorage.getItem(key); if(t) root.dataset.theme=t;}catch(e){}
  document.getElementById("theme").onclick=()=>{
    const dark=root.dataset.theme?root.dataset.theme==="dark":matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme=dark?"light":"dark"; try{localStorage.setItem(key,root.dataset.theme)}catch(e){}
  };
  document.getElementById("menu").onclick=()=>document.body.classList.toggle("nav-open");
  const links=[...document.querySelectorAll("#nav a")];
  links.forEach(a=>a.addEventListener("click",()=>document.body.classList.remove("nav-open")));
  const byId=Object.fromEntries(links.map(a=>[a.hash.slice(1),a]));
  const obs=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(l=>l.classList.remove("active"));byId[e.target.id]?.classList.add("active");}})},{rootMargin:"-30% 0px -65% 0px"});
  document.querySelectorAll("main section[id]").forEach(s=>obs.observe(s));
  const bar=document.getElementById("progress");
  addEventListener("scroll",()=>{const h=document.documentElement;bar.style.width=(100*h.scrollTop/(h.scrollHeight-h.clientHeight))+"%"},{passive:true});
  document.querySelectorAll("pre").forEach(p=>{const b=document.createElement("button");b.className="copy";b.textContent="Copy";
    b.onclick=()=>{navigator.clipboard?.writeText(p.innerText.replace(/Copy$/,"").trim());b.textContent="Copied";setTimeout(()=>b.textContent="Copy",1200)};p.appendChild(b)});
  document.querySelectorAll("main table").forEach(t=>{if(!t.parentElement.classList.contains("table-wrap")){const w=document.createElement("div");w.className="table-wrap";t.before(w);w.appendChild(t)}});
})();
</script>
</body>
</html>
"""


def main() -> int:
    parts = [p.read_text(encoding="utf-8") for p in sorted((GUIDE / "sections").glob("*.html"))]
    body = "\n".join(parts)
    found = re.findall(r'<section\s+id="([^"]+)"\s+data-title="([^"]+)"', body)
    nav = []
    for i, (sid, title) in enumerate(found):
        nav.append(f'<a href="#{sid}"><span class="n">{NUMERALS[i % len(NUMERALS)]}</span>{html.escape(html.unescape(title))}</a>')
        body = re.sub(rf'(<section\s+id="{re.escape(sid)}"[^>]*>)',
                      rf'\1\n<div class="chapter">Chapter {NUMERALS[i % len(NUMERALS)]}</div>', body, count=1)
    out = TEMPLATE.replace("__NAV__", "\n".join(nav)).replace("__SECTIONS__", body).replace("__REPO__", REPO)
    (GUIDE / "index.html").write_text(out, encoding="utf-8")
    print(f"docs/guide/index.html: {len(found)} sections from {len(parts)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
