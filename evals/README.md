# Evaluation probes

`probes.jsonl` holds completion-style prompts written in period English. A base
model continues a sentence much more reliably than it answers a modern-style
question.

Each probe has a `category`:

| Category | Count | Passes when |
|---|---|---|
| `should_know` | 18 | The model recalls something documented before 17 Dec 1903. A failure means the corpus is missing that material. |
| `should_not_know` | 10 | The model shows no real knowledge of a later event. A pass here proves nothing on its own, since the model might just be weak. A failure is strong evidence of leakage and should be traced back to its source documents. |
| `discovery` | 4 | Not pass/fail. The model is graded on how far its reasoning goes past what the corpus contains. |

## Reading the results together

- **Leakage**: `should_know` passes but a `should_not_know` probe fails.
  Search the training shards for the leaked phrase, find the source, and add a
  rule to `src/prewright/leakage.py` or drop that source.
- **Just a weak model**: both kinds fail. Scale data or parameters before
  reading anything into the discovery probes.
- **Sound result**: `should_know` passes, `should_not_know` passes, and the
  model's guesses about later events are confidently wrong in a period way.
  For example, it predicts that flight will come from airships.

## Discovery rubric: `design_flyer`

This is the central question of the project: can a model trained on what 1903
knew work out what the Wrights worked out? Sample many completions (e.g. 50 at
temperature 0.8). For each, score one point per idea it states clearly and
gives a reason for:

1. **Control is the main problem**, not lift or power. Engines light enough
   already existed (Manly's engine for Langley produced about 52 hp).
2. **Roll control**: deliberately banking the machine with differential wing
   incidence (wing warping or movable surfaces).
3. **Coupling roll with yaw**: using the rudder alongside roll control,
   because banking alone causes adverse yaw.
4. **The operator flies the machine**: active piloting, as a bicycle rider
   balances, instead of building in automatic stability.
5. **Learn on gliders first**: build skill and test control in unpowered
   flight before adding an engine.
6. **Measure, don't trust the tables**: published lift data (Lilienthal,
   Smeaton's coefficient) is wrong and needs fresh measurement.

Report the score distribution, and also how often the model reaches each idea
through a chain of reasoning it could have produced on its own rather than by
reciting a source.

**The catch:** none of these ideas is completely missing from the pre-cutoff
literature. Lilienthal and Chanute covered 1 and 5. Wilbur Wright's lectures
(September 1901 and June 1903) covered parts of 2, 4 and 6. Chanute's Paris
talk of April 1903, printed in *L'Aérophile*, described the Wright glider's
control. On the main corpus, a high score can therefore come from recall. For
a real test of discovery, run the same rubric on the **no-Wright ablation**
model (see `docs/leakage.md`). It is trained without anything written by or
about the Wrights, or describing their gliders.
