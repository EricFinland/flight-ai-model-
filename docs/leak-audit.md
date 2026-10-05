# Flight-scrub leak audit (red team)

Corpus audited: `/home/user/data/corpus/{main,control}/train.jsonl` as built on
2026-10-05 (stats.json: 16,168 main train books; 12,541 paragraphs cut from kept
books; 241 books dropped as flight books). Tooling: `scripts/audit_leaks.py`,
patterns in `evals/redteam_patterns.txt`, ripgrep anchors in
`evals/redteam_anchors.txt`.

## Phase 1 verdict

**Verdict: NEEDS FIX.** The fix is small and targeted. The lexicon works well
on vocabulary: no `balloon`, `aeronaut` or `flying machine` survives. What gets
through is (1) one OCR/old spelling of Montgolfier, (2) a few future-tech coinages
the lexicon lacks (`air-car`, `air-boat`, `ship of the air`), and (3) passages
that describe flight in words the lexicon doesn't cover ("wings", "fly", "up-borne
by the air"), where only a heading or a neighbouring paragraph hit and the
±1-paragraph window stopped short of the rest.

Method: a single ripgrep pass of 12 targeted red-team regexes over the first 56%
of main train (9,542 of 16,168 books, 3.76 of 6.66 GB; the rest was still
running at the deadline), 6,788 matches, every suspicious group read by hand.

### Real leaks found (category a, policy violations) in 56% of main

| # | book (main line) | text that survives | why the lexicon missed it |
|---|---|---|---|
| 1 | Holmes, *Medical Essays* (1038) and *Complete Works* (1266) | "Like the famous toy of **Mongolfier**, it rose by means of heated air … it collapsed and fell." | spelling `Mongolfier` (no t) |
| 2 | Erasmus Darwin, *The Botanic Garden* pt 2 (3693) | "Rife, great **MONGOLFIER**! urge thy venturous flight / High o'er the Moon's pale ice-reflected light" | same |
| 3 | Johnson, *Rasselas* (two editions) | Ch. VI body: "the mechanist … He that can swim needs not despair to fly … You will be necessarily up-borne by the air if you can renew any impulse upon it faster than the air can recede" | only the chapter title ("art of flying") hit, and the window is ±1 paragraph |
| 4 | Rostand, *Cyrano de Bergerac* (408) | "Six novel methods, all, this brain invented!" and the verse listing ways to rise to the moon (dew flasks, rarefied air, smoke in a globe, a saltpetre-fired spring machine…) | the Archytas/Regiomontanus paragraph hit; the following paragraphs were outside the window |
| 5 | Emerson, *English Traits* (9533) | Bacon: "…to make machines, which, by means of a suit of wings, should fly in the air in the manner of birds." But the secret slept with Bacon. | no lexicon word |
| 6 | Bellamy, *Equality* (2740) | "the invention of the **air-car** …", "chartering an **air car** this afternoon … a bird's-eye view of the city" | `air-car` not in lexicon |
| 7 | Shiel, *The Purple Cloud* (3925) | "fixed upon the first old green **air-boat** that seemed fairly sound … tested and prepared my ship" | `air-boat` not in lexicon |
| 8 | *From Farm House to the White House* (5860), publisher's ad for *Motor Rangers* (also post-1903) | "their **ship of the air** is attacked by huge birds of the air" | `ship of the air` |
| 9 | Barr, *A Prince of Good Fellows* (6681) | "He has constructed a pair of wings, and each soldier being provided with them can sail through the air across the Border." | no lexicon word |
| 10 | *Prairie Farmer* 1884 (9178) | "Every one travels through the air on wings of electricity" (a prophecy) | no lexicon word |
| 11 | *Cambrian Sketch-Book* (9408) | "We shall travel through the air, which we use to propel us along … The thing was discovered by our wise men" | no lexicon word |

Ambiguous (c): Pardoe, *Thousand and One Days* (a stranger builds a flying chest
with wings, wheels and springs: a mechanical fairy tale); Calmet, *Phantom World*
("Mangonia, where there were vessels which were borne through the air", Agobard's
cloud-ships legend); Lucian, *Trips to the Moon* ("my wings were a kind of
Daedalian contrivance").

Allowed (b), the large majority of matches: "flew/sailing/carried through the
air" (projectiles, birds, witches, fairies, djinn: about 300 matches, all
innocent apart from #9-11); Daedalus (213) and Icarus (163), myth; "attempt at
flight", "idea of flight", "flight was impossible", "we will/can/shall fly"
(fleeing, about 250, all innocent); `aerial *` (1,681: aerial perspective,
roots, spirits, currents, all innocent); `silk bag`, `oiled silk` (dress,
surgery, all innocent); `flying chariot/car` (Ramayana, Ovid, myth; Napoleon's
"flying chariot" is a lift); `air-vessels` (botany, physiology). Name-based
probes were noise: Sadler, Cocking, Gambetta, Hargrave, Wenham, du Temple,
Pilcher, Mouillard, Rozier and Stringfellow are all other people or places in
the surviving text (for example Audubon's partner Rozier, Wenham Lake ice, the
Rue du Temple). Glaisher, Coxwell and Nadar survive only in non-flight senses
(the astronomer, a cork, Casal Nadar on Malta, a photographer named in Verne).

### Rate

About 11 real-leak passages, roughly 14 paragraphs, in about 9,500 books and
about 20 M paragraphs. That's **about 0.7 leaked paragraphs per million**, or
about **12 per 10k books**. The bar was below about 1 per 10k books, so this
fails by roughly 10x. All of the leaks are incidental (a sentence or a
paragraph) except #3 and #4, which are full descriptions of attempted flight.
The scrub removed about 12.5k paragraphs plus 241 books. What is left is about
0.1% of the original flight signal, but it includes the two classic literary
descriptions of human-powered and lighter-than-air flight (Rasselas, Cyrano).

### Recommended regex additions to `src/prewright/flight.py` (exact)

Add these as unconditional `Term`s unless a context is given, with anchors in
TERM_ANCHORS:

```python
Term("mongolfier", _rx(r"\bmongolfier(?:e|es|s)?\b|\bmontgolf\w*|\bmongolfiere?s?\b")),   # anchors: "mongolf", "montgolf"
Term("air_vehicle", _rx(r"(?<!compressed )(?<!compressed-)\bair[- ]?(?:cars?|boats?|chariots?)\b|\bships?\s+of\s+the\s+(?:air|sky|skies|clouds)\b")),  # anchors: "air-car","air car","aircar","air-boat","air boat","airboat","air-chariot","air chariot","of the air","of the sky","of the clouds"
Term("wings_machine", _rx(r"\b(?:machines?|engines?|contrivances?|apparatus)\b[^.]{0,80}\bwings?\b[^.]{0,80}\bfly\b"
                          r"|\b(?:constructed|made|invented|contrived|fashioned)\s+(?:him(?:self)?\s+|them(?:selves)?\s+)?(?:a\s+)?(?:pair|suit|set)\s+of\s+wings\b"
                          r"|\bsuit\s+of\s+wings\b|\bdespair\s+to\s+fly\b|\bup-?borne\s+by\s+the\s+air\b")),  # anchors: "wings", "despair to fly", "borne by the air"
Term("air_travel_tech", _rx(r"\b(?:travel\w*|sail\w*|journey\w*|voyag\w*|transport\w*|navigat\w*)\s+(?:\w+\s+){0,3}?through\s+the\s+air\b"),
     _rx(r"\b(?:machine\w*|invent\w*|discover\w*|propel\w*|electric\w*|engines?|apparatus|contrivances?)\b"),
     note="magic/witch travel through the air has none of these words, so it is kept"),  # anchors: "through the air"
```

Granularity: the Rasselas and Cyrano leaks come from the ±1-paragraph window.
I recommend `NEIGHBOURS = 2` (or a forward-biased window of 1 back, 3 forward),
or adding the Cyrano trigger `\bsix\s+novel\s+methods\b` and the Rasselas lines
above as patterns. The `wings_machine` term covers Rasselas; Cyrano needs the
wider window or its own pattern.

Estimated main paragraphs these would remove: about 40 hit paragraphs (about
25 for `air_vehicle`, including about 10 fairy-tale air-boats and the botanical
"air-vessel" being excluded, about 6 for `mongolfier`, about 5 for
`wings_machine`, about 5 for `air_travel_tech`), so about **120 paragraphs**
with ±1 neighbours. That is about 1% of what is already removed. Exact counts
are in the Phase 2 tables below.
