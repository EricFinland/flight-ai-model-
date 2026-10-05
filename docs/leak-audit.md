# Flight-scrub leak audit (red team)

Two passes over the scrubbed `main` corpus. Phase 1 (first auditor, 56% of
`main`, the first build) found 11 survivors and led to five new lexicon terms
and a re-scrub. Phase 2 (second auditor) checks the re-scrubbed corpus in full,
and in particular the 1,415 books the model actually trains on. Tooling:
`scripts/audit_leaks.py` (usage at the top of the file), patterns in
`evals/redteam_patterns.txt`, ripgrep anchors in `evals/redteam_anchors.txt`,
pattern counts from the Phase 2 run in `evals/leak_audit/counts.json`.

## Bottom line (for the README)

> **Phase 2 verdict: the vocabulary scrub works, and descriptions of flight
> without flight vocabulary still get through.** No lexicon word (`balloon`,
> `aeronaut`, `flying machine`, ...) survives anywhere in `main`, and all 11
> Phase 1 leaks are gone. In the 1,415 books the model trained on, we found
> **7 books with passages that describe human flight in other words, about
> 120 paragraphs in all**, most of them in one book:
>
> - L. Frank Baum, *The Master Key* (1901): a boy flies around the world with
>   an electric "traveling machine" strapped to his wrist ("He turned the
>   indicator of his traveling machine and immediately rose into the air"),
>   about 100 paragraphs. A second edition of the same book was dropped as a
>   flight book; this one has too few lexicon hits to be dropped.
> - Rostand, *Cyrano de Bergerac* (prose translation): Cyrano's six ways to
>   reach the moon ("Since smoke persists in rising, I might have blown into a
>   globe enough of it to carry me up!"), about 10 short paragraphs.
> - Lagerlöf, *Gösta Berling*: the inventor Kevenhüller "began to fasten wings
>   to his shoulders" on his tower and "lay and floated high above the
>   earth", 6 paragraphs.
> - One sentence or one paragraph each: Erasmus Darwin's "unconquer'd steam
>   ... bear / The flying chariot through the fields of air"; "M. Gay Lussac,
>   in the year 1804, ascended to the height of 23,000 feet"; Voltaire's
>   mechanic Simon who "made himself wings; he flew; and, like Icarus, he
>   fell"; "the engineer of the aerial steam-ship".
>
> That is about 67 per million trained paragraphs (120 of 1.78 M) and 7
> books in 1,415 (about 49 per 10k books). Without *The Master Key* it is
> about 20 paragraphs. These counts are a lower bound: they come from regex
> searches plus a hand review of the text around every cut in the trained
> books, not from reading 1,415 books. The same kinds of passage occur
> elsewhere in `main` (tower-jumpers with home-made wings, flying automata,
> the Darwin couplet in four more books). About 10% of the scrub's cuts are
> false positives, and at least 7 of the 245 dropped books have nothing to do
> with flight (a villain named Hargrave, Simon Renard, the Ballon d'Alsace).
> The fixes below would apply to future builds; the current training run
> uses the corpus as audited here.

## Phase 1 (historical: the first build, kept as written)

Corpus audited: `/home/user/data/corpus/{main,control}/train.jsonl` as built on
2026-10-05 (stats.json: 16,168 main train books; 12,541 paragraphs cut from kept
books; 241 books dropped as flight books). Tooling: `scripts/audit_leaks.py`,
patterns in `evals/redteam_patterns.txt`, ripgrep anchors in
`evals/redteam_anchors.txt`.

### Phase 1 verdict

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

## Phase 2: after the fix

**Verdict: NEEDS FIX (for future builds).** The Phase 1 fix worked. No flight
vocabulary survives, and none of the 11 Phase 1 passages is left. But 7
trained books still describe human flight in ordinary words (about 120
paragraphs, ~100 of them in *The Master Key*). That is far above the bar of
under 1 per 10k books. The fixes in section 7 are small. Two structural ones
(same-title drops, a character-based window) and four terms would remove
every trained leak found here. The current training run cannot pick them up.

### What was audited

- `main/{train,val}.jsonl` as re-derived from `control` by
  `scripts/rescrub_corpus.py` with the 48-term lexicon (written 12:17 on
  2026-10-05; 16,165 train + 323 val books, 245 flight books dropped, 12,725
  paragraphs cut from kept books). `control/{train,val}.jsonl` for comparison.
- **The trained subset.** `scripts/tokenize_corpus.py` orders train books by
  `sha1("prewright-1903:<id>")` and takes those with rank <= the cutoff in
  `/home/user/data/tokenized/main/train.meta.json`
  (`17208c1d8c0770b1c8179362ed6a14338472e5f6`). That gives exactly **1,415
  main books** (1.78 M paragraphs, 150 M tokens). The tokenized files were
  written at 12:27, after the re-scrub, so the model trains on the corpus as
  audited here. The same cutoff gives 1,436 control books: the same 1,415
  plus 21 flight books that main dropped.
- The SFT chat data (`/home/user/data/sft/{train,val}.jsonl`, 11,997 pairs).

### Method

1. **Phase 1 strings.** `p1_*` patterns for the exact text of the 11 Phase 1
   leaks, plus the `fix_*` copies of the new lexicon terms.
2. **Full red-team scan.** 109 patterns (`lex_`, `fix_`, `p1_`, the first
   auditor's `rt_`/`name_`/`idea_`/`mod_`, and 35 new `rt2_` patterns; the
   `fix2_` patterns were added later),
   streamed over every paragraph of main and control in one process:

       python3 scripts/audit_leaks.py \
           $DATA/corpus/main/train.jsonl $DATA/corpus/main/val.jsonl \
           --compare $DATA/corpus/control/train.jsonl $DATA/corpus/control/val.jsonl \
           --with-trained --out evals/leak_audit --samples 12 --keep 250

   It took 77 minutes at `nice 19` beside the two training runs, using under
   150 MB. In the trained subset every hit of the specific patterns was read.
   For the broad ones (`rt_through_air_travel`, `rt_aerial_any`,
   `rt_aero_any`, `rt_daedalus`, `rt_icarus`, `rt2_cleave_air`,
   `rt2_air_compound`, `rt2_rise_from_earth`, `rt2_mount_into_air`,
   `name_early_experimenters`), only the hits that also contain a
   technology word were read. `rt_cyrano` (the name, 1,525 times) and the
   `mod_` patterns were not read. In the rest of main,
   the 1,501 hits of the high-signal patterns were filtered to the 360 with
   both a technology word and a flight word, and those were read.
3. **Cut edges.** For each of the 322 blocks cut from the 183 trained books
   that lost paragraphs, the four kept paragraphs on each side were dumped
   and the ones with flight cues were read. This found the Cyrano, Gösta
   Berling and Darwin leaks, which no pattern targets.
4. **Italics.** Project Gutenberg marks italics with underscores, and `_`
   counts as a word character, so `\b`-anchored terms miss `_flying
   machine_`. Main was re-matched with `_` replaced by a space.
5. **Precision.** 30 random paragraphs from
   `flight_removed_train_sample.jsonl`, the 110 non-`balloon` cut blocks in
   the trained books, and the hit terms and contexts of 95 of the 245
   dropped flight books.

### 1. The Phase 1 leaks are gone

| # | Phase 1 leak | in main now | evidence |
|---|---|---|---|
| 1 | Holmes, "toy of Mongolfier" | gone | `p1_mongolfier` 0 (control 228) |
| 2 | Darwin, "Rife, great MONGOLFIER!" | gone | same |
| 3 | *Rasselas* ch. VI, the mechanist | gone | `p1_rasselas` 1 hit, an unrelated "mechanist ... imperfect imitations" of joints |
| 4 | Cyrano's six methods, verse edition (1254) | gone | `p1_cyrano` 0; the whole 18-paragraph scene is cut. But see the prose edition (41949) below |
| 5 | Emerson, Bacon's "suit of wings" | gone | `p1_suit_of_wings` 0; *English Traits* is now dropped whole (3 of 445 paragraphs hit) |
| 6 | Bellamy, *Equality*, "air-car" | gone | `p1_air_car_boat` 0 (control 55) |
| 7 | Shiel, *The Purple Cloud*, "air-boat" | gone | same |
| 8 | Motor Rangers ad, "ship of the air" | gone | same |
| 9 | Barr, "constructed a pair of wings" | gone | `p1_suit_of_wings` 0 |
| 10 | *Prairie Farmer*, "travels through the air on wings of electricity" | gone | `p1_travel_air_wise` 6 left, all "news flew on the wings of electricity" (telegraph) or the soul in Andersen (allowed) |
| 11 | *Cambrian Sketch-Book*, "We shall travel through the air" | gone | same |

The two `fix_wings_machine` hits left in main are Virgil's "in despair to fly
away", which the term was written to keep.

### 2. Red-team patterns, main vs control

Paragraphs (books) per pattern. Main counts are after the scrub; control is the
same books before it. The judgment column covers the hits in main and is
based on the reads described under Method.

| pattern | main, all 16,488 books | main, 1,415 trained books | control, all 16,733 books | control, 1,436 trained books | judgment of the main hits |
|---|---:|---:|---:|---:|---|
| `lex_balloon` | 0 (0) | 0 (0) | 6,854 (2,015) | 511 (143) | sanity: 0 in main (control 6,854) |
| `lex_aeronaut` | 0 (0) | 0 (0) | 1,274 (366) | 97 (35) | sanity: 0 in main |
| `lex_flying_machine` | 0 (0) | 0 (0) | 386 (177) | 32 (15) | sanity: 0 in main |
| `fix_mongolfier` | 0 (0) | 0 (0) | 225 (89) | 14 (10) | 0 in main: fix works |
| `fix_air_vehicle` | 0 (0) | 0 (0) | 62 (27) | 2 (2) | 0 in main: fix works |
| `fix_wings_machine` | 2 (2) | 0 (0) | 15 (15) | 0 (0) | 2 = Virgil 'in despair to fly away' (allowed, as intended) |
| `fix_air_travel_tech` | 0 (0) | 0 (0) | 42 (40) | 2 (2) | 0 in main: fix works |
| `fix_cyrano_methods` | 0 (0) | 0 (0) | 1 (1) | 1 (1) | 0 in main; but see Cyrano prose edition (41949) below |
| `rt_balloon_ocr` | 6 (5) | 0 (0) | 83 (42) | 23 (2) | 6 main, 0 trained: OCR/dialect noise (allowed) |
| `rt_through_air_travel` | 1,068 (826) | 117 (84) | 1,216 (911) | 131 (90) | birds, missiles, witches, fairies, djinn (allowed) except The Master Key (real) |
| `rt_in_air_car` | 84 (75) | 7 (7) | 110 (94) | 7 (7) | fairy cars, trains, baskets; Stockton's negative-gravity basket (ambiguous) |
| `rt_person_rose_air` | 32 (32) | 4 (4) | 37 (37) | 4 (4) | fairies, birds, arrows (allowed) |
| `rt_ascent_gas` | 28 (27) | 2 (2) | 89 (63) | 7 (6) | manure gas, colliery gas, solar hydrogen (allowed) |
| `rt_hydrogen_envelope` | 27 (11) | 1 (1) | 53 (29) | 2 (2) | chemistry recipes (allowed) |
| `rt_silk_bag` | 167 (139) | 10 (9) | 184 (152) | 14 (12) | purses, wig bags (allowed) |
| `rt_inflated_gas` | 16 (14) | 3 (3) | 90 (55) | 8 (6) | veterinary bloat, a goldbeater-skin simile (allowed) |
| `rt_ballast_grapnel` | 0 (0) | 0 (0) | 10 (4) | 0 (0) | 0 in main |
| `rt_air_vessels` | 157 (64) | 26 (4) | 185 (80) | 30 (7) | lungs, seaweed, pumps (allowed) |
| `rt_flying_variants` | 230 (197) | 18 (17) | 269 (220) | 21 (20) | flying boats (whaleboats), coaches, Erasmus Darwin's steam 'flying chariot' (real, 47314) |
| `rt_wings_machine` | 83 (81) | 6 (6) | 133 (113) | 8 (8) | windmills, insects, fishing flies; tower-jump attempts outside trained set (real) |
| `rt_man_wings` | 44 (41) | 4 (4) | 51 (48) | 5 (5) | Daedalus, Wayland (myth, allowed); Voltaire's Simon the mechanic (real, 35623) |
| `rt_steerable` | 2 (2) | 0 (0) | 5 (5) | 0 (0) | 2, not flight |
| `rt_navigate_air` | 3 (3) | 1 (1) | 50 (36) | 9 (4) | 'navigable or not' (allowed) |
| `rt_propeller_air` | 18 (14) | 0 (0) | 32 (23) | 1 (1) | 0 trained; ship screws, a satire (allowed) |
| `rt_kite_man` | 3 (3) | 0 (0) | 4 (4) | 0 (0) | 0 trained |
| `rt_parachute_ocr` | 93 (55) | 11 (5) | 623 (277) | 72 (27) | Strabo's Parachoathras, parasols (allowed) |
| `rt_aerial_any` | 3,147 (1,670) | 302 (149) | 4,244 (1,954) | 385 (173) | aerial perspective/course/grace (allowed); 'engineer of the aerial steam-ship' (real, 43586) |
| `rt_aero_any` | 566 (296) | 58 (30) | 2,132 (643) | 190 (62) | aerolites, Aeropus, aerophane (allowed) |
| `rt_aviat` | 2 (2) | 0 (0) | 32 (14) | 0 (0) | 0 trained |
| `rt_flight_machine` | 20 (20) | 2 (2) | 41 (35) | 3 (3) | fleeing + machine (allowed); The Master Key (real) |
| `rt_invent_fly` | 101 (94) | 6 (6) | 155 (132) | 12 (11) | fly-shuttle, fly-wheel; Wren's 'thirty ways of flying', Campanella 'discovered how to fly' outside trained set (real) |
| `rt_cyrano` | 1,592 (45) | 1,525 (8) | 1,667 (49) | 1,537 (9) | the name (allowed); the speech itself is the leak |
| `rt_rasselas` | 7 (5) | 0 (0) | 9 (7) | 0 (0) | 0 trained |
| `rt_peter_wilkins` | 220 (73) | 7 (7) | 230 (80) | 8 (8) | Peter Wilkins's flying people (fantasy, allowed) |
| `rt_bacon_flying` | 24 (22) | 3 (2) | 30 (27) | 3 (2) | Nathaniel Bacon's flight = fleeing (allowed) |
| `rt_wilkins_flying` | 30 (25) | 1 (1) | 43 (36) | 2 (2) | Peter Wilkins (allowed) |
| `rt_lana` | 1 (1) | 0 (0) | 8 (6) | 0 (0) | 0 trained |
| `rt_fiction_flyers` | 108 (84) | 21 (10) | 365 (106) | 22 (11) | Quercus robur, 'robur et aes triplex' (allowed) |
| `rt_daedalus` | 618 (321) | 74 (37) | 651 (341) | 75 (37) | myth (allowed); Lucian's 'Daedalian contrivance' (ambiguous) |
| `rt_icarus` | 362 (230) | 31 (21) | 399 (254) | 34 (23) | myth (allowed) |
| `name_nadar` | 25 (17) | 0 (0) | 66 (33) | 1 (1) | 0 trained |
| `name_glaisher_coxwell` | 76 (39) | 2 (2) | 308 (71) | 3 (3) | Glaisher the astronomer/editor (allowed) |
| `name_green_aeronaut` | 8 (7) | 0 (0) | 40 (25) | 4 (2) | 0 trained; Mr. Green's ascent of Mount Earnslaw (allowed) |
| `name_vauxhall_ascent` | 0 (0) | 0 (0) | 22 (13) | 2 (2) | 0 in main |
| `name_pilatre_rozier` | 71 (26) | 1 (1) | 139 (52) | 4 (3) | Rozier's medical journal (allowed) |
| `name_charles_robert` | 10 (10) | 3 (3) | 18 (16) | 3 (3) | brothers Colling, Innes, Hamilton (allowed) |
| `name_gambetta` | 10 (7) | 1 (1) | 21 (15) | 2 (2) | Gambetta at Tours, no balloon (allowed) |
| `name_siege_paris_post` | 1 (1) | 0 (0) | 1 (1) | 0 (0) | 1 main, not flight |
| `name_early_experimenters` | 1,931 (512) | 96 (42) | 2,476 (542) | 101 (44) | Fort Pilcher, Rue du Temple, Jane Wenham, Hargrave (allowed noise) |
| `name_maxim_machine` | 26 (16) | 2 (2) | 33 (21) | 2 (2) | Maxim gun, a maxim (allowed) |
| `name_whitehead_fly` | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 |
| `name_langley` | 0 (0) | 0 (0) | 4 (3) | 0 (0) | 0 |
| `name_blanchard_channel` | 1 (1) | 0 (0) | 36 (17) | 1 (1) | 0 trained |
| `name_sadler` | 3 (1) | 0 (0) | 6 (4) | 1 (1) | 0 trained |
| `name_gay_lussac_biot` | 3 (3) | 2 (2) | 28 (15) | 4 (3) | 'M. Gay Lussac, in the year 1804, ascended to the height of 23,000 feet' (real, 47748); Biot the town |
| `name_cocking` | 2 (2) | 0 (0) | 12 (9) | 0 (0) | 0 trained |
| `name_coutelle_fleurus` | 12 (12) | 3 (3) | 31 (21) | 3 (3) | battle of Fleurus, Coutelle the Egyptologist (allowed) |
| `idea_men_fly` | 20 (19) | 2 (2) | 24 (23) | 2 (2) | 'a man may fly from' = flee (allowed) |
| `idea_soar_man` | 9 (9) | 0 (0) | 9 (9) | 0 (0) | 0 trained |
| `idea_problem_flight` | 42 (42) | 3 (3) | 49 (47) | 3 (3) | 'hope of flight' = escape (allowed) |
| `idea_flying_possible` | 3 (3) | 1 (1) | 6 (6) | 1 (1) | Utopia, fleeing (allowed) |
| `idea_learn_to_fly` | 182 (148) | 15 (13) | 196 (158) | 16 (14) | birds, 'fly the filth of sin' (allowed) |
| `idea_fly_like_bird` | 60 (53) | 7 (7) | 65 (58) | 7 (7) | boats, Daedalus myth, a fairy ring (allowed) |
| `idea_wings_for_man` | 9 (9) | 0 (0) | 10 (10) | 0 (0) | 0 trained |
| `mod_wright` | 9,232 (2,154) | 662 (180) | 9,310 (2,184) | 671 (185) | surname (not flight) |
| `mod_einstein` | 556 (111) | 65 (8) | 562 (113) | 65 (8) | the word 'relativity' |
| `mod_gutenberg` | 1,291 (574) | 132 (51) | 1,299 (580) | 132 (51) | transcriber notes (leakage filter, not flight) |
| `mod_year_1900s` | 9,406 (2,733) | 728 (233) | 9,823 (2,796) | 762 (238) | mostly 1900-1902; see the 1920s play ads in 41739 |
| `p1_mongolfier` | 0 (0) | 0 (0) | 228 (92) | 14 (10) | 0: Phase 1 leaks #1-2 gone |
| `p1_rasselas` | 1 (1) | 0 (0) | 3 (3) | 0 (0) | 1 = Paley-style 'mechanist ... imperfect imitations' of joints (allowed); Rasselas gone |
| `p1_cyrano` | 0 (0) | 0 (0) | 3 (1) | 3 (1) | 0 for the verse edition (1254) |
| `p1_suit_of_wings` | 0 (0) | 0 (0) | 4 (4) | 0 (0) | 0: Bacon (Emerson) and Barr gone |
| `p1_air_car_boat` | 0 (0) | 0 (0) | 55 (22) | 2 (2) | 0: Bellamy, Shiel, Motor Rangers ad gone |
| `p1_travel_air_wise` | 6 (6) | 2 (2) | 8 (8) | 2 (2) | 6 = telegraph/soul 'on the wings of electricity' (allowed); Prairie Farmer and Cambrian gone |
| `rt2_rise_from_earth` | 476 (439) | 34 (30) | 494 (454) | 34 (30) | lifting people/objects, Antaeus (allowed) |
| `rt2_mount_into_air` | 290 (262) | 37 (28) | 337 (293) | 38 (29) | birds, fairies, myths; The Master Key x9 (real) |
| `rt2_cleave_air` | 314 (260) | 31 (27) | 326 (271) | 32 (27) | arrows, birds, Clavileno in Don Quixote (allowed) |
| `rt2_artificial_bird` | 228 (170) | 21 (13) | 263 (184) | 21 (13) | Andersen's clockwork nightingale, fishing flies (allowed) |
| `rt2_winged_car` | 133 (115) | 8 (8) | 135 (117) | 8 (8) | sailing ships, Marvell (allowed) |
| `rt2_vessel_air` | 63 (59) | 6 (5) | 87 (72) | 7 (6) | fairy flying ship, mirage; The Master Key (real) |
| `rt2_navigate_skies` | 1 (1) | 0 (0) | 2 (2) | 1 (1) | 0 trained |
| `rt2_flying_man` | 236 (210) | 18 (17) | 263 (223) | 19 (18) | fleeing soldiers, Peter Wilkins (allowed) |
| `rt2_levitate` | 127 (60) | 22 (9) | 132 (63) | 23 (9) | spiritualist levitation of mediums (allowed: not flight technology) |
| `rt2_lighter_air_machine` | 18 (17) | 3 (3) | 37 (30) | 4 (4) | owls, gossamer spiders, a seance chair (allowed) |
| `rt2_hydrogen_lift` | 54 (33) | 4 (3) | 142 (69) | 13 (7) | chemistry, the sun (allowed) |
| `rt2_hot_air_bag` | 16 (16) | 3 (3) | 52 (38) | 3 (3) | medical baths, stoves (allowed) |
| `rt2_gas_silk` | 50 (42) | 3 (2) | 110 (74) | 9 (5) | gilding recipes, oiled silk dressings (allowed) |
| `rt2_wings_fasten` | 90 (83) | 12 (11) | 103 (93) | 13 (12) | Thumbelina, Navajo dancers, butterflies; Kevenhuller 'began to fasten wings to his shoulders' (real, 56158) |
| `rt2_flight_apparatus` | 21 (21) | 3 (3) | 37 (36) | 3 (3) | fleeing + engine (allowed) |
| `rt2_daedalus_machine` | 46 (34) | 9 (7) | 56 (41) | 9 (7) | Daedalus inventor of sails (allowed); Lucian (ambiguous) |
| `rt2_voyage_moon` | 132 (94) | 14 (13) | 171 (114) | 20 (16) | idioms, Lucian, Ariosto, Mahomet (allowed) |
| `rt2_projectile_moon` | 1,548 (110) | 5 (5) | 1,586 (115) | 5 (5) | Verne's projectile: 1,548 main paragraphs in 110 books, but 0 of Verne in the trained set (5 = 'Columbiad' poem, Gun Club titles) |
| `rt2_air_compound` | 135 (84) | 26 (9) | 205 (112) | 33 (14) | 'sky-pilot' = chaplain (allowed) |
| `rt2_ether_ship` | 57 (44) | 3 (3) | 137 (56) | 36 (6) | gravity in physics (allowed) |
| `rt2_war_in_air` | 45 (40) | 4 (3) | 48 (43) | 4 (3) | Milton, Lucian, legend (allowed) |
| `rt2_bombs_from_above` | 5 (5) | 1 (1) | 9 (8) | 1 (1) | rhetoric (allowed) |
| `rt2_ocean_of_air` | 24 (24) | 1 (1) | 31 (31) | 1 (1) | insect flight (allowed) |
| `rt2_bird_machine` | 46 (40) | 3 (3) | 483 (248) | 44 (21) | 'flying engine' = locomotive, flying coaches (allowed) |
| `rt2_wings_material` | 77 (71) | 9 (8) | 78 (72) | 9 (8) | 'silken wings' of love, canvas sails (allowed); Farrow's silk-and-whalebone wing machines outside trained set (real) |
| `rt2_tower_jump` | 3 (3) | 0 (0) | 3 (3) | 0 (0) | 0 trained |
| `rt2_legend_fliers` | 107 (56) | 6 (5) | 125 (68) | 11 (7) | Bladud as founder of Bath, Simon Magus (legend, allowed) |
| `rt2_umbrella_descent` | 36 (36) | 3 (3) | 44 (42) | 3 (3) | 0 relevant |
| `rt2_height_feet_ascent` | 113 (79) | 6 (6) | 143 (91) | 8 (7) | mountains, geysers; Gay-Lussac (real, see above) |
| `rt2_birds_eye_view_we` | 14 (14) | 1 (1) | 19 (19) | 2 (2) | forest floor (allowed) |
| `rt2_aeronaut_names2` | 90 (49) | 14 (4) | 159 (76) | 17 (7) | Monck Mason the historian, a Professor Lowe, Meusnier the encyclopaedist (allowed) |
| `rt2_foreign_flight` | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 in main and control |
| `rt2_rocket_carry` | 3 (2) | 0 (0) | 3 (2) | 0 (0) | 0 trained |
| `rt2_man_bird_imitate` | 3 (3) | 0 (0) | 3 (3) | 0 (0) | 0 trained |
| `rt2_air_conquer` | 20 (20) | 3 (3) | 28 (26) | 3 (3) | 'empire of the air' of eagles (allowed) |

`lex_*` = 0 everywhere in main is the main result: the lexicon removes its own
vocabulary completely. Nearly every other hit in main is ordinary English that
the policy keeps (fleeing, birds, myth, fairy tales, chemistry, spiritualism).
The new `rt2_*` patterns found no new flight vocabulary: zero hits for German
or French flight words, `sky-pilot` only as slang for a chaplain, and Verne's
projectile (1,548 paragraphs in 110 main books, kept by policy as a cannon
shot) does not occur in the trained subset at all.

### 3. Residual leaks: passages that describe human flight

**In the 1,415 trained books** (every one read in context):

| book (id) | what survives | paragraphs | why the scrub missed it |
|---|---|---:|---|
| Baum, *The Master Key* (45347) | A boy flies with an electric "traveling machine" on his wrist, which has an up/down indicator and a compass: "He turned the indicator of his traveling machine and immediately rose into the air"; he crosses the Atlantic, hovers over London, and lifts a cannibal chief by the heels. The plot of the book. | ~100 | Only 3 blocks hit (balloon, air-ship, "travel through the air"). Words like "rose into the air", "flight" and "traveling machine" are not in the lexicon, and 3 hits in 1,110 paragraphs is below the 0.5% book rule. The other edition (436) has 5 hits in 966 paragraphs and was dropped. |
| Rostand, *Cyrano de Bergerac*, prose translation (41949) | Act III: "To learn how I got up there? Easily. Through an invention of mine", "I invented six different ways of assaulting the virgin blue!", rarefied air in a cedar box heated by mirrors "to lift me", "a machinist and an artificer" riding a steel grasshopper fired by saltpetre, "I might have blown into a globe enough of [smoke] to carry me up!", an iron plate chasing a magnet so "the ascension" lasts indefinitely. | ~10 | `cyrano_methods` matches the verse translation only. Only the Archytas/Regiomontanus line hit, and in a play every speaker name is its own paragraph, so the ±1-paragraph window removed one line. |
| Lagerlöf, *Gösta Berling* (56158) | Kevenhüller, "inventor of the automatic carriage" (the "flying-machine" in that sentence was cut): "began to fasten wings to his shoulders", "got his wings on and set out ... he was out in the air. He lay and floated high above the earth", "If he had only been able to make such a pair of wings for every one of them!" | 6 | The hit paragraphs ("flying-machine") are 2 to 4 paragraphs away. |
| Jackson, *Philosophy of Natural Theology* (47314) | Erasmus Darwin: "Soon shall thy arm, unconquer'd steam! afar / Drag the slow barge, or drive the rapid car. / Or on wide-waving wings expanded bear / The flying chariot through the fields of air." (a prophecy of steam-powered flight) | 1 | No lexicon word. The neighbouring balloon paragraph is 2 away. The couplet also survives in 4 untrained books (31359, 35916, 46216, 46659). |
| *The Reason Why* (47748) | "M. Gay Lussac, in the year 1804, ascended to the height of 23,000 feet." | 1 | A one-line Q&A entry with no lexicon word. |
| Voltaire, *Philosophical Dictionary* vol. 3 (35623) | "a very skilful mechanic named Simon, who not only made things fly across the stage ... He made himself wings; he flew; and, like Icarus, he fell." | 1 | "made himself wings" is not covered by `wings_machine`. |
| *Popular Account of the Manners and Customs of India* (43586) | "...to the engineer of the aerial steam-ship..." | 1 sentence | `aerial_craft` has "aerial steam-carriage" and "aerial ship" but not "aerial steam-ship". |

Ambiguous, counted as allowed: Lucian, *Trips to the Moon* (10430): in
*Icaromenippus*, Menippus cuts the wings off an eagle and a vulture, calls them
"a kind of Daedalian contrivance", and flies to the moon. This is classical
satire in the Daedalus tradition, and the footnote naming Bishop Wilkins's
"Art of Flying" was cut. Alger, *Helen Ford* (53868) and a *Harper's* story
(36516): the "flying machine" paragraphs were cut, and what remains is "my
father's invention" and "my machine was under consideration", with no flight
content. *Occultism and Common-Sense* (36730): mediums "levitated" at séances.
A *Phaedrus* footnote (25512): "_Flying-machine_" (the italics hid it from the
lexicon) for the stage crane that lowered the gods.

**Elsewhere in main** (not trained, found by the filtered read; not
exhaustive):

| book (id) | what survives |
|---|---|
| Farrow, *The Missing Prince* (45291) | "The machines ... consisted of two large silk and whalebone wings, fastened on to the back with straps ... by flapping one's arms about it was possible to fly quite comfortably." |
| *Yorkshire Oddities* (47386) | "He also constructed for himself a pair of wings ... his attempt to fly from the mast-top of a boat in the Humber failed." |
| William of Malmesbury's *Chronicle* (50778) | Eilmer: "He had by some contrivance fastened wings to his hands and feet..." (he is not named "Elmer of Malmesbury", so `malmesbury_flier` misses him) |
| *Some of our East Coast Towns* (53890) | John Gainsborough "invented wings, by means of which he essayed to fly" |
| *Cameos from English History* (9474) | "a foolish attempt to fly from the top of a tower with wings of his own contrivance" |
| *Darkness and Dawn* (47825) | "The broad wings of the machine upbore for a moment the light form of Nazarius as he sprang from the tower" |
| *Automata Old and New* (55817) | Regiomontanus's iron fly and "artificial eagle which ... flew to meet the Emperor Maximilian"; a toy driven by "a light screw propeller ... two fixed wings" |
| *Memoirs of Robert-Houdin* (42916), *Curious Facts* (41625), *Ethics of Diet* (55785) | Montroyal's (Regiomontanus's) iron fly; Archytas "inventor of the wooden flying dove" |
| Aubrey, *Brief Lives* (47787) | Wren "invented thirty severall wayes of flying" |
| Lombroso, *The Man of Genius* (50539) | Campanella's City of the Sun: "they have discovered how to fly" |
| *London in Modern Times* (35084) | a bubble company "for making flying engines" |
| Erasmus Darwin couplet | 4 more books (see above) |
| Stockton, *A Chosen Few* (25549) | "negative gravity" lifts a basket "high in the air" (ambiguous: anti-gravity fiction) |

### 4. Rates, with caveats

| | books | real-leak books | per 10k books | real-leak paragraphs | per million paragraphs |
|---|---:|---:|---:|---:|---:|
| trained subset | 1,415 | 7 | ~49 | ~120 (~20 without *The Master Key*) | ~67 (~11) |
| all of main (lower bound) | 16,488 | 20 or more | 12 or more | ~150 or more | 7 or more |

- The trained subset was searched much harder than the rest of main: all of
  its hits and the text around all of its cuts were read. The all-main row
  counts only what the filtered read found. The real rate in main is probably
  close to the trained rate, which would mean a few hundred passages.
- These rates are higher than Phase 1's (about 12 per 10k books) because the
  method is different, not because the fix made anything worse. Phase 1 only
  read pattern hits. *The Master Key*, the Cyrano prose edition, Gösta Berling
  and the Darwin couplet were all in the first build too.
- The bar was below 1 per 10k books, so this still fails by an order of
  magnitude. What is left is different in kind, though. No flight word
  survives, and Phase 1's balloon, aeronaut and air-car vocabulary is gone.
  The residue is narrative: one fantasy novel about electric flight, one
  speech, one tower flight, and a handful of single sentences. A model
  trained on it can still meet the *idea* of a man rising into the air by a
  machine, mostly in Baum's book.
- Every number here comes from regex recall plus reading. A passage that
  shares no word with any of 109 patterns and lies more than 4 paragraphs
  from a cut is invisible to this audit.

### 5. Precision of the scrub

**Removed paragraphs.** 30 random paragraphs from
`flight_removed_train_sample.jsonl` (seed 2026; 13 from dropped flight books,
17 from passages cut from kept books):

| judgment | count | examples |
|---|---:|---|
| real flight (ascents, balloons, aeronauts, flying machine) | 17 | Verne, J. M. Bacon, Ashton, Holmes's *Mongolfier*, Alger's flying machine |
| by-policy hit, no flight content ("balloon" in another sense, figurative) | 10 | "swelling up like a balloon", balloon sleeves, soap balloons, "balloon guineas", the Balloon Society, "coiffure Montgolfier", Port Balloon, "up in a balloon" (an idiom, twice) |
| **false positive against the policy** | 3 | *Gassies v. Ballon* (a surname); a shell "sailing through the air" (`air_travel_tech`, Civil War); Faust's magic "aerial voyages" (magic is kept) |

So about **10% of hit paragraphs are false positives under the policy** (95%
CI roughly 2 to 27%), and another third carry no flight information but are
removed by design ("balloon" in any sense). Each hit also costs its two
neighbours.

**Cuts in the trained books.** Of the 322 cut blocks in the 1,415 trained
books, 212 come from the `balloon` family (by policy). Of the other 110, about
28 are false positives under the policy: Simon Renard (*Mary I*, twice) and Renard the fox
(*Phaedrus*); Langley the squire in *Digby Heathcote* (twice, triggered by
"fly", a carriage) and a Norfolk place; Garibaldi's comrade Zambeccari
(twice); Colonel Pilcher in the Boer War; "aerial voyage(r)" for martins,
hummingbirds and a secretary-bird (`aerial_craft` has no bird exception);
"navigation of the air" for birds; Breguet's "parachute" shock-guard in
watches; Astolpho's hippogriff, a fairy
"travelled through the air", Mrs. Guppy's spiritualist "aerial transit",
Casanova's blanket-tossing "aerial journeys"; "aeromotor" (a windmill);
Marey's *Terrestrial and Aerial Locomotion* in publishers' lists. That is
about 9% of all cuts. Another ~12 are "parachute" for gliding animals and
seeds (flying phalangers and lemurs, lizards, dandelions), which follows the
policy but is zoology.

**Dropped flight books.** The hit terms of 95 of the 245 were checked (all
the low-density drops and every title that did not look like flight). **Seven
are false positives, with no flight content at all:**

| book | why it was dropped |
|---|---|
| Brunton, *Self-control* (41196) | the villain Colonel Hargrave: 31 `flight_experimenters` hits, context "flew to the table", "flight" |
| Ainsworth, *The Tower of London* (49850) | Simon Renard: 27 hits ("ascended to the roof", "flight") |
| Lanzi, *History of Painting in Italy* vol. V (39996) | the Zambeccari gallery in Bologna: 5 hits (`zambeccari` has no context) |
| Mungo Park, *Travels in the Interior of Africa* vol. 2 (5305) | "baloon" = a Mandingo hut ("permitted me to sleep in a large baloon"); vol. 1 is kept |
| Belloc, *The Path to Rome* (7373) | the Ballon d'Alsace, a mountain: 12 hits |
| Marryat, *Borneo and the Indian Archipelago* (26844) | the town of Baloongan: 8 hits |
| Michell, *The Art and Practice of Hawking* (48437) | a hawk named Parachute: 10 hits |

About 14 more are by-policy drops with no human-flight content: "balloon" as
a chemist's receiver (*Chymistry*, 1777, six years before the Montgolfiers),
a boiler type (*Steam Boiler Explosions*), a balloon frame (*Woodward's
Country Homes*), a bloated colon (*Intestinal Ills*), a flask (*Perfumery*),
a simile (*Great Tornado*), toys and dresses (*Bits About Home Matters*,
*Verdant Green*); "parachute" for seeds and gliding animals (*Seed
Dispersal*, *Common Objects of the Microscope*, *Dragons of the Air*). *The
Time Machine* is dropped for 3 figurative balloons in 322 paragraphs. The rest
of the 95 (Verne, Wells, Poe, ballooning chapters, siege-of-Paris memoirs,
Moffett, Rotch, Glaisher readers, boys' balloon stories) are correct. No
"Flying Dutchman" or "flight = fleeing" title was dropped: the title rule
fired only on the six titles with *Balloon*, *Flight*, *Manflight* or
*Tissandier*.

### 6. SFT chat data

`/home/user/data/sft/{train,val}.jsonl`: 0 of 11,997 pairs hit the lexicon, and
none of their source books is a dropped or filtered book. The full pattern set
hits only an "aerial body" (spirits), an "Anthony Henson", and two Q&A pairs
from Verne's *From the Earth to the Moon* ("Is it possible to transmit a
projectile up to the moon?"), which is a cannon shot and allowed by policy.
**Clean.**

### 7. Recommended fixes for `src/prewright/flight.py`

These are not applied. The training run started at 12:28 on the audited
corpus, so they would only affect future builds (re-derive with
`scripts/rescrub_corpus.py`, then re-tokenize). Each regex was checked against
the leaks above and a set of innocent sentences, and is in
`evals/redteam_patterns.txt` as `fix2_*`, so the audit can count them. Main
hit counts are given where measured.

Recall, new `Term`s (anchors in brackets):

```python
# 1. Cyrano, prose translation  [six different, virgin blue, cedar box, into a globe, steel-legged, thrown it up]
Term("cyrano_prose", _rx(r"\bsix\s+different\s+ways\s+of\s+assaulting\b|\bassaulting\s+the\s+virgin\s+blue\b"
     r"|\brarefied\s+the\s+air\s+in\s+a\s+cedar\s+box\b|\bblown\s+into\s+a\s+globe\b|\bsteel-legged\s+grasshopper\b"
     r"|\bmagnet\s+and\s+thrown\s+it\s+up\s+into\s+the\s+air\b")),
# 2. home-made wings (Kevenhuller, Eilmer, tower-jumpers, Voltaire's Simon; also Daedalus/Wayland retellings,
#    consistent with the existing "artificial wings" rule). Not "made themselves wings": riches do that (Prov. 23:5).  [wing]
Term("wings_fastened", _rx(r"\b(?:fasten\w*|strap\w*|tied|bound|buckled|fixed)\s+(?:on\s+)?(?:\w+\s+){0,2}?wings?\s+(?:to|on|upon)\s+"
     r"(?:his|her|their|my|your|the)\s+(?:shoulders|arms|back|body|hands)\b|\bgot\s+(?:his|her)\s+wings\s+on\b"
     r"|\b(?:constructed|made|invented|contrived)\s+(?:for\s+)?(?:him|her)self\s+(?:a\s+)?(?:pair|suit|set)\s+of\s+wings\b"
     r"|\bmade\s+(?:him|her)self\s+wings\b|\binvented\s+wings\b|\bwings\s+of\s+(?:his|her|their)\s+own\s+contrivance\b"
     r"|\bcontrivance\s+fastened\s+wings\b")),
# 3. Erasmus Darwin's steam "flying chariot" (not the first line, which is about railways)  [fields of air, realms of air, waving wings]
Term("steam_flying_chariot", _rx(r"\bflying[- ]chariot\s+through\s+the\s+(?:fields|realms)\s+of\s+air\b|\bwide[- ]waving\s+wings\s+expanded\b")),
# 4. The Master Key  [traveling machine, travelling machine]
Term("travelling_machine", _rx(r"\btravell?ing\s+machines?\b"),
     _rx(r"\b(?:air|fl(?:y|ew|ying|ight)|soar\w*|rose|rise|indicator)\b")),
# 5. Gay-Lussac and Biot's 1804 ascents, sentence-level  [lussac, biot]
Term("gay_lussac_ascent", _rx(r"\b(?:gay[- ]lussac|biot)\b[^.]{0,120}\bascen(?:ded|t|sion)\b[^.]{0,80}\b(?:height|feet|metres|altitude)\b")),
# 6. more flying automata  [flying dove, iron fly, iron eagle, artificial fly, artificial eagle]
Term("flying_automata_2", _rx(r"\bwooden\s+flying\s+dove\b|\b(?:iron|artificial)\s+(?:fly|eagle)\b"),
     _rx(r"\bwooden\s+flying\s+dove\b|\bregiomontanus|\bmuller\b|\bmontroyal|\barchytas|\bnuremberg|\bmaximilian")),
```

and in `aerial_craft`, change `ships?` to `(?:steam[- ]?)?ships?` ("aerial
steam-ship").

Measured on main (`--only fix2_`; the counts for 2, 3 and 5 are for the
regexes as revised here, read off the hits of the first versions):

| term | main paragraphs (books) | trained | what it hits |
|---|---:|---:|---|
| `cyrano_prose` | 5 (1) | 5 (1) | the Cyrano speech only |
| `wings_fastened` | ~17 (~14) | ~5 (4) | Gösta Berling, Voltaire, Eilmer, Yorkshire, East Coast, Cameos; Daedalus, Wayland, *Kalevala*, Andersen retellings (myth, consistent with the "artificial wings" rule) |
| `steam_flying_chariot` | 4 (4) | 1 (1) | the couplet; an earlier draft that also matched "unconquer'd steam" hit 6 railway quotations |
| `travelling_machine` | 14 (2) | 13 (1) | *The Master Key*, plus 1 false positive (a coach in a history of inventions) |
| `gay_lussac_ascent` | 1 (1) | 1 (1) | *The Reason Why*; the first draft, a context term, hit 19, mostly Biot's acoustics |
| `flying_automata_2` | 9 (7) | 0 | Regiomontanus's eagle and fly, Archytas's dove (by policy, like `flying_automata`) |
| `aerial_craft` + steam-ship | 1 (1) | 1 (1) | the India "aerial steam-ship" |

A broader "machine word + rose/floated/flew into the air" term was tried and
rejected: 173 paragraphs in 150 books, nearly all innocent (smoke, kites,
séances, catapults).

With these terms *The Master Key* would be dropped whole: its 3 existing hit
paragraphs plus 13 `travelling_machine` paragraphs come to about 16 of 1,110
(1.4%), above the 0.5% book rule.

Structural (more effective than more regexes):

7. **Window by characters, not paragraphs.** In plays every speaker name is a
   paragraph, so ±1 paragraph can be 20 characters (Cyrano 41949). Grow each
   window until at least ~400 characters are removed on each side
   (`MIN_NEIGHBOUR_CHARS`).
8. **Same work, same decision.** When a book is dropped as a flight book, drop
   kept books with an identical title (ignoring case and "Complete"). Today
   that would catch exactly two: *The Master Key* (45347, trained) and *Nuts
   and Nutcrackers* (31685).
9. **Underscore italics.** In `normalize()`, replace `_` with a space before
   matching. `\b` fails between `_` and a letter, so `_Flying-machine_`,
   `_Montgolfier_` and `_aerial navigation_` are invisible to every
   `\b`-anchored term. This is rare in practice: 25 paragraphs in all of main,
   of which only ~4 are flight (the Phaedrus footnote, two *Yellow Fairy Book*
   "Flying Ship" titles, Le Conte's flying-machine analogy). The rest would be
   new false positives (lace "ballon", "chasse au renard").

Precision (fixes for the false positives above):

10. `flight_experimenters`: the context `\bfly\w*|\bflight|\bflew\b|ascen[dt]\w*|\bmachine`
    lets novels with a character named Hargrave or Renard through ("flew to",
    "flight" = fleeing). Use
    `_rx(r"\baerial\b|\bglid\w*|\bsoar\w*|\bkites?\b|\bflying[- ]machine|\baeroplane|\bwings?\b[^.]{0,60}\b(?:machine|apparatus)")`
    instead. That would have saved *Self-control* and *The Tower of London*.
11. `langley`: drop `\bfly\w*` (the carriage) and `\bmodel\b` from the context:
    `_rx(r"\baerodrome|\bsoar\w*|\baerial\b|\bpotomac\b|\bflying[- ]machine|\bplanes?\b")`.
12. `zambeccari`: make it a context term,
    `_rx(r"ascen\w*|\baerial\b|\bgas\b|\bballoon")` (Bologna's Zambeccari
    family and Garibaldi's comrade are not the aeronaut).
13. `balloon`: replace `\bbaloon\w*` with `\bbaloons?\b(?!gan)` plus a context
    of ascent, gas or air (Mungo Park's huts, Baloongan). Add
    `(?!\s+d['’]alsace|\s+de\s+(?:guebwiller|servance))` after `ballon`, and
    `(?<!v\. )` before it (the Vosges summits; *Gassies v. Ballon*).
14. `aerial_craft`: `voyag\w*|journey\w*|excursions?` fire on birds and insects.
    Require that no `\b(?:bird|martin|swallow|swift|insect|humming)` occurs in
    the same sentence, or move these three words to a context term needing a
    human or machine word.
15. `air_travel_tech`: exclude shells and projectiles, e.g. add
    `(?<!shell )(?<!shot )`, or require that the context word is not "engine"
    in a battle paragraph. This is low priority: one false positive in the
    sample.

### 8. Other findings (not flight)

- *A Christmas Carol* by C. Z. Barnett (41739, **trained**) ends with Samuel
  French play advertisements from the 1920s and 30s (*The Bat*, *Skidding*,
  *Louder, Please*, and a "Criterion Pictures" film comedy). The "airplanes"
  paragraph was cut by the flight lexicon, but the rest is post-1903 text the
  leakage filter missed.
- `mod_gutenberg`: 132 trained paragraphs (51 books) still have transcriber
  notes ("The following is a list of changes made to the original").
  Harmless, but not period text.

### Reproducing

```bash
DATA=/home/user/data
# everything (77 min, 1 core, <150 MB; writes summary.md, counts.json, samples_*.jsonl, hits_*.jsonl)
python3 scripts/audit_leaks.py $DATA/corpus/main/train.jsonl $DATA/corpus/main/val.jsonl \
    --compare $DATA/corpus/control/train.jsonl $DATA/corpus/control/val.jsonl \
    --with-trained --out evals/leak_audit --samples 12 --keep 250
# only the Phase 1 strings and the fixes, trained books only (a few minutes)
python3 scripts/audit_leaks.py $DATA/corpus/main/train.jsonl --trained-only --only p1_,fix_,fix2_
```

`evals/leak_audit/counts.json` holds the per-pattern counts of the Phase 2
run. The hit and sample files (`*.jsonl`) are gitignored, because they quote
the control corpus.
