# The showcase page

`index.html` in this folder is the public page for the project: one self-contained file (inline CSS, the
two plots embedded as data URIs, Google Fonts as the only external request, with a Georgia fallback). It is
generated; do not edit it by hand.

## Rebuild it

After the evaluation, the training curves and the chat showcase have been written to `evals/results/`:

```bash
python3 scripts/build_site.py --results evals/results --data /home/user/data --out docs/site/index.html
```

Inputs: `evals/results/{eval.json, curve.json, chat_showcase.md, flight_gap_curve.png, loss_curves.png}` and
`<data>/corpus/stats.json`, `<data>/corpus/manifest.csv`, `<data>/tokenized/{main,control}/train.meta.json`.
Numbers are formatted by the same code as `scripts/fill_readme_results.py`, so the page and the README
agree. A figure whose input is missing is left out, never filled with a placeholder; the script lists what
it left out on stderr. It refuses a `--fake` evaluation unless given `--allow-fake`, which also stamps the
page as a test build. The page builds in well under a second and loads no model.

## Publish it with GitHub Pages

1. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch**.
2. Branch **main**, folder **/docs**, then **Save**.

The site is then served from `https://ericfinland.github.io/flight-ai-model-/`. Publishing `/docs` makes
the whole `docs/` folder the site root, so:

- `docs/index.html` is a one-line redirect to `site/`, where this page lives;
- `docs/.nojekyll` turns off GitHub's Jekyll step, which would otherwise try to render the design
  documents in `docs/` as pages (and could fail on template-like syntax in them). They are served as plain
  `.md` files instead; the page links to their rendered versions on github.com.

The page's own address is `https://ericfinland.github.io/flight-ai-model-/site/`. Every push to `main`
that changes `docs/` republishes it within a minute or two.
