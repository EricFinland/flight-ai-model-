# The flight scrub

`src/prewright/flight.py` removes every trace of human flight from the `main`
corpus. `control` is the same corpus without this step. This page records the
policy, why the lexicon looks the way it does, how much text is cut, and what
the scrub is known to miss. `docs/PLAN.md` is the source of truth for the
policy. This page explains how it is implemented.

## Policy

**Removed**: real or plausible human flight and flight technology. That
includes attempts, proposals, and fiction that treats flight as technology
(Verne's *Robur*, Poe's balloon hoax, boys' adventure stories with air-ships):

- balloons of every kind, toys included (a toy balloon still shows that a gas
  bag can lift), plus air, fire and hydrogen balloons, ballooning,
  balloonists, aerostats and aerostation, Montgolfières and Charlières;
- aeronauts and aeronautics, airships, dirigibles (the noun, and "dirigible
  balloon"), Zeppelin, flying machines and flying ships, aeroplanes,
  airplanes, aerodromes, gliders and gliding machines, soaring machines,
  ornithopters, helicopters, parachutes, man-lifting kites, aviation and
  aviators, airmen;
- aerial navigation and the aerial ship, vessel, machine, car, voyage,
  journey, fleet, ascent, excursion, locomotion and so on; "navigation of the
  air", "conquest of the air", "art of flying", "mechanical/artificial/human
  flight", "artificial wings", "heavier-than-air" machines;
- Tennyson's *Locksley Hall* prophecy ("airy navies", "argosies of magic
  sails", "pilots of the purple twilight");
- the names of aeronauts and experimenters: Montgolfier (also *Mongolfier*),
  Lunardi, Garnerin, Santos-Dumont, Chanute, Kitty Hawk, Pilâtre de Rozier,
  Zambeccari, Tissandier, Gusmão, Francesco Lana, Oliver of Malmesbury, the
  Wright brothers, and the flying automata of Archytas and Regiomontanus;
- added after the red-team audit ([leak-audit.md](leak-audit.md)):
  future-fiction vehicles (Bellamy's *air-car*, Shiel's *air-boat*, "ships of
  the air"), machines with wings to fly and a "suit of wings", "travel through
  the air" by machine or electricity, and Cyrano's methods for rising to the
  moon.

**Kept**, because ordinary English needs it: birds, bats and insects flying;
*flight* meaning fleeing; flights of stairs and arrows; flying colours, flying
fish, flying buttresses, flying squadrons; myth, magic and fairy tale (Icarus,
Daedalus, angels, witches, Pegasus, Laputa, flying carpets); "lighter/heavier
than air" as plain physics of gases; Congreve rockets; kites as children's
toys and kites the birds; the word *maxim* and the Maxim gun; *aerial* in its
other senses (aerial roots, aerial perspective, "aerial flight" of a bird).

## How matching works

1. **Normalisation.** Each paragraph is matched on a normalised copy:
   accents folded (`aérostat` → `aerostat`, `Pilâtre` → `Pilatre`), ligatures
   expanded (`æronaut` → `aeronaut`), soft hyphens dropped, words joined
   across hyphenated line breaks (`bal-\nloon` → `balloon`,
   `air-\nship` → `airship`), and whitespace collapsed. PG-19 text is
   hard-wrapped at about 70 characters, so phrases often span lines.
2. **Lexicon.** `TERMS` is a tuple of 48 named regexes. Most terms always count.
   Names and words with a common innocent sense are **context terms**: they
   count only if a flight-context word (ascent, aerial, flying, wings,
   machine, soaring...) appears in the same paragraph. The context terms are
   Blanchard (Laman Blanchard and Blanchard Jerrold are writers), Langley (an
   astronomer, and a place-name), Maxim (a common word, and a gun),
   Glaisher/Coxwell (J. W. L. Glaisher was also a mathematician), Cayley
   (Arthur Cayley the mathematician), Lilienthal (a homoeopath and a rabbi of
   the name), gas-bag (also slang for a windbag), minor experimenters
   (Henson, Stringfellow, Wenham, Pilcher, Ader, Hargrave, Giffard, Renard,
   Krebs, Nadar, Andrée...) and "travel through the air" (only with a
   machine, invention, engine or electricity word; witches keep flying). That
   makes 9 context terms and 39 unconditional ones.
3. **Speed.** Each term has lowercase *anchors*, literals that every match
   must contain. A `str.find` scan over the whole book picks out the
   paragraphs that can match, and the regexes run only on those. A test
   checks that the anchors cover the recall examples.
4. **Reuse.** `FLIGHT_PATTERN` (all unconditional terms in one regex),
   `contains_flight(text)`, `find_flight_hits(text)` and `TERMS` are public,
   so the audit and evaluation code can share the lexicon.

## Granularity

- A paragraph with a hit is dropped together with **one paragraph on each
  side**. The neighbours usually carry the set-up and the aftermath ("the car
  rose rapidly", "we came down in a turnip field") without a lexicon word.
- **Bridging**: if two hit paragraphs are at most 4 paragraphs apart,
  everything between them goes too. A chapter on ballooning is cut as one
  block instead of leaving its innocent-looking middle paragraphs behind.
- Blocks longer than 4,000 characters (books without blank lines) are split
  at indented lines, or every 12 lines, before scrubbing, so one hit cannot
  take a whole chapter with it.
- **Whole-book drop**: a book is dropped from `main` if
  - its title matches the lexicon (*Five Weeks in a Balloon*, *Wonderful
    Balloon Ascents*), or
  - more than **0.5%** of its paragraphs hit, with at least 3 hit paragraphs,
    or
  - it has **30 or more** hit paragraphs, however long it is.

  The thresholds were tuned on a 1,500-book sample. At 0.5% the books
  dropped are the ones *about* flight or with a flight episode at their
  centre: Verne's *Five Weeks in a Balloon* (13%), *Robur the Conqueror*
  (16%) and *The Mysterious Island* (2%), Moffett's *Careers of Danger and
  Daring* (7%, with a balloonist chapter), Wells's *First Men in the Moon*,
  and boys' adventure stories with balloon escapes. A flat "8 or more hits"
  rule would also have dropped large compendia with a dozen scattered
  mentions: the collected works of Oliver Wendell Holmes (12 of 13,912
  paragraphs), a dictionary of quotations (9 of 34,729), the *Challenger*
  Radiolaria report (10 of 18,769), Brassey's *Yachting* (balloon sails).
  Paragraph cutting with bridging handles these well. Losing them would cost
  more general knowledge than it removes flight.

## Judgment calls

- **"balloon" in any sense is removed**: balloon sails (balloon jib, balloon
  topsail), balloon sleeves, balloon frames in carpentry, "ballooning"
  spiders, chemists' balloon flasks. These are rare, and the word itself is
  the main thing the model must not learn.
- French/German *ballon*, *Luftschiff*, *dirigeable* and *aéronef* are
  removed wherever they appear in English books.
- **"Artificial wings"** is removed even in a Daedalus retelling. The myth
  survives in other wording, and real tower-jumpers (Oliver of Malmesbury,
  Bacqueville, Rasselas's artist) used the same phrase.
- **Fairy-tale flying ships** ("The Fool of the World and the Flying Ship")
  are removed, because the phrase is also the usual name for an airship in
  fiction.
- **Space-travel fiction** with no flight words (Verne's projectile, Tucker's
  *Voyage to the Moon*) is kept. It is not flight through the air, and the
  plan does not list it. *First Men in the Moon* is dropped anyway: its
  balloons and parachutes push it over the density threshold.
- The leakage filter's soft rule for "airplane" can push a book into REVIEW,
  which drops it from **both** corpora. This is rare and harmless.

## Known gaps

- Descriptions of flight that use no lexicon word and are more than one
  paragraph from a hit: "the car rose above the clouds", "he was carried up
  into the air by the machine". Bridging and the neighbour rule catch most of
  them. The red-team audit below found about 11 survivors in the part of the
  corpus it searched; they are now covered, but the search was not
  exhaustive.
- Spelling variants outside the lexicon, such as OCR errors that split a word
  with a space (`bal loon`) or garble letters (`baiIoon`). PG-19 text is
  proofread, so these are rare.
- Bishop Wilkins's flying chariot and Erasmus Darwin's "flying-chariot"
  couplet are not covered by name. (Cyrano's six ways to reach the moon now
  are, by `cyrano_methods`.)
- Context terms trade recall for precision. "Mr. Coxwell rose from Wolverton"
  is kept unless a context word is present.
- Foreign-language passages about flight in other words (e.g. German
  *Flugmaschine*).

## Red-team checks on the scrubbed `main` text

Searches over a 250 MB slice of `main/train.jsonl` for near misses: "fly/
flying through the air", "ascended in/from", "men will/could fly", "wings
to his arms/shoulders", "inflate", "sail through the air", "navigate the
sky", "aerial ...", Vauxhall ascents, Archytas, Bladud, Wilkins. The matches
were knives and snow "flying through the air", hill ascents, "we will fly"
in the sense of flee, and angels with wings. Cyrano's "pigeon of Archytas"
was the only real miss in that search, and it is now in the lexicon.

A later, wider red-team pass ([leak-audit.md](leak-audit.md)) did find about 11
survivors in 56% of `main`: the *Mongolfier* spelling, *air-car*, *air-boat*
and "ship of the air", Bacon's "suit of wings", Rasselas's mechanist and
Cyrano's moon machines, and "travel through the air" by machine. They are now
the `montgolfier` (widened), `air_vehicle`, `wings_machine`, `air_travel_tech`
and `cyrano_methods` terms, and `main` was re-derived from `control` with
`scripts/rescrub_corpus.py`. The re-scrub (48 lexicon terms, 9.5 minutes on
2 workers) changed the corpus as follows; the statistics in the next section
are those of the current, re-scrubbed corpus:

| | first build | after re-scrub |
|---|---:|---:|
| flight books dropped whole (train / val) | 241 (237 / 4) | 245 (240 / 5) |
| main books kept (train / val) | 16,168 / 324 | 16,165 / 323 |
| paragraphs cut from kept books | 12,541 | 12,725 |
| kept books with at least one paragraph cut | 2,439 | 2,458 |
| paragraphs with a lexicon hit | 9,976 | 10,085 |
| held-out flight paragraphs (`flight_eval.jsonl`) | 495 | 497 |
| main/train characters | 6,577,221,840 | 6,576,274,316 |

Matches of the new terms: `montgolfier` 279 (was 265, the difference being
*Mongolfier*), `air_vehicle` 68, `air_travel_tech` 43, `wings_machine` 14,
`cyrano_methods` 6.

## Corpus statistics

From the full build of 17,350 PG-19 books dated 1902 or earlier
(`python3 scripts/build_corpus.py`, 16 min with 3 workers), after the
re-scrub. The numbers are in `/home/user/data/corpus/stats.json` and
`manifest.csv`.

| Decision | Train | Val | Total |
|---|---:|---:|---:|
| kept | 16,165 | 323 | 16,488 |
| dropped_flight_book | 240 | 5 | 245 |
| dropped_leakage | 176 | 5 | 181 |
| dropped_review | 335 | 8 | 343 |
| dropped_nonenglish | 59 | 1 | 60 |
| dropped_empty | 33 | 0 | 33 |

`dropped_flight_book` books are dropped from `main` only; they stay in `control`. The other drops apply to both corpora.

| Corpus | Split | Books | Characters | Est. tokens (chars/4) |
|---|---|---:|---:|---:|
| main | train | 16,165 | 6,576,274,316 | 1,644,068,579 |
| main | val | 323 | 117,419,501 | 29,354,875 |
| control | train | 16,405 | 6,676,461,588 | 1,669,115,397 |
| control | val | 328 | 119,001,225 | 29,750,306 |

- `main` train is 1.50% smaller than `control` train (100,187,272 characters).
- Paragraphs cut from kept books (hits, neighbours and bridged gaps): 12,725.
- Paragraphs containing a lexicon hit (kept and dropped books): 10,085.
- Held-out flight evaluation set (`flight_eval.jsonl`, hit paragraphs from validation books): 497 paragraphs.
- `flight_removed_train_sample.jsonl`: 500 paragraphs for auditing.
- Flight books: 245 (6 by title, 239 by hit density).
- Kept books with at least one paragraph cut: 2,458.

Top lexicon terms by number of matches, over every book that passed the date, leakage and English filters (kept books and flight books):

| Term | Matches |
|---|---:|
| balloon | 9,772 |
| aeronaut | 1,500 |
| parachute | 706 |
| aerial_craft | 670 |
| aerostat | 536 |
| flying_machine | 478 |
| airship | 463 |
| flight_experimenters | 327 |
| montgolfier | 279 |
| glaisher | 247 |
| art_of_flying | 174 |
| navigation_of_the_air | 126 |
| blanchard | 96 |
| garnerin | 86 |
| pilatre_de_rozier | 76 |
| air_vehicle | 68 |
| tissandier | 68 |
| lunardi | 56 |
| langley | 46 |
| air_travel_tech | 43 |
| aviation | 32 |
| zambeccari | 30 |
| airy_navies | 25 |
| zeppelin | 21 |
| santos_dumont | 20 |
| maxim | 19 |
| dirigible | 18 |
| glider | 16 |
| flying_automata | 16 |
| soaring_machine | 15 |

Then wings_machine, conquest_of_the_air and cayley 14 each, gas_bag 12,
lilienthal and helicopter 11, chanute 10, kitty_hawk 8, cyrano_methods and
airman 6.

Term names are the `Term.name` values in `flight.py`. `balloon` covers every
spelling and compound, and `aeronaut` includes the `aero-` compounds
(`aeronef`, `aeroplane`, `aerodrome`).

**Audit.** On the first build, re-running `find_flight_hits` over the
finished `main` corpus found **0** hits in all 324 validation books and 0 in
the first 1,932 training books (814 M characters) checked. For the current
corpus, the evaluation's corpus pass (`scripts/evaluate.py`, section
`corpus`) audits every trained book of both corpora; its counts go into the
README (`audit_main_hits`, `audit_control_hits`).

**Cost of the density rule.** At 0.5%, a 500-paragraph book with three
figurative balloons ("as loose as balloons") is dropped whole. That is
deliberate, since recall comes first. All 245 flight books together are
1.4% of `control`'s characters (93.9M of 6.80B), far more text than one
training run uses. If precision matters more later, raise
`BOOK_FRACTION_MIN_HITS` to 5.

**Other filters**, for context: `dropped_review` is almost always the
leakage filter's post-1903-year rule, and it catches mis-dated books (for
example a 1962 history catalogued as 1869). `dropped_nonenglish` holds
French, Italian, German and Finnish books, plus directories and taxonomic
lists with almost no running English. `dropped_empty` holds image-only
volumes.

Outputs (`/home/user/data/corpus/`): `{main,control}/{train,val}.jsonl` (one
book per line: `id`, `title`, `year`, `text`; paragraphs separated by blank
lines, wrapped lines joined, verse-like blocks keep their line breaks),
`flight_eval.jsonl` and `flight_removed_train_sample.jsonl` (`id`, `title`,
`source` = `passage`|`flight_book`, `text`), `manifest.csv`, `stats.json`.
Validation is the ~2% of books whose `sha1("pg19:<id>")` falls in the first
20 of every 1,000 values. It is the same for both corpora.
