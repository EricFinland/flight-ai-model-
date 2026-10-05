# Model card: Prewright (flight-free 1902 language model)

Four checkpoints, all in `checkpoints/` as fp16 (16-bit) files of about 17 MB, written at the end
of the build (`scripts/finish_pilot.sh`; until it finishes, some of them are missing). To talk
to the model, use `main_chat.pt`; the base models only continue text. Terms are explained in the
README's [Glossary](README.md#glossary).

| file | what it is |
|---|---|
| `main.pt` | the flight-free base model: pretrained on English books published in 1902 or earlier, with all human-flight material removed |
| `control.pt` | the control: identical in every respect except that the flight material was left in |
| `main_chat.pt` | `main.pt` fine-tuned to answer questions, on period question-and-answer text |
| `control_chat.pt` | `control.pt` fine-tuned the same way on the same data, for side-by-side chat comparisons |

Project overview and results: [README.md](README.md). Design decisions: [docs/PLAN.md](docs/PLAN.md).

Numbers marked `TBD` are filled in from the evaluation by `scripts/fill_readme_results.py`.

## Intended use

- **Research and demonstration.** The pair of base models is a controlled experiment: what does a
  language model know, and how does it behave, when one well-defined concept (human flight) is
  absent from everything it has read? Compare `main` with `control`, always at matched settings.
- **Curiosity and teaching.** Talking to `main_chat.pt` shows what a 1902 reader without any idea
  of flight might say, in the voice of the period.
- **A template.** The pipeline (date filter, concept scrub, matched control, paired probes) can
  be reused for other concepts or cutoffs, and scaled up with the GPU configs.

**Not intended** for any practical task: it is far too small to be reliable, its knowledge is a
century out of date, and it reproduces the prejudices of its sources. Do not use it for advice,
facts, or text shown to people who have not been told what it is.

## Training data

- **Source**: PG-19 (DeepMind's Project Gutenberg books published before 1919), the only large
  pre-1903 source reachable from the build machine. Books whose PG-19 `publication_date` is 1902
  or earlier: 17,350.
- **Filters applied to both models**: Gutenberg boilerplate stripped; an anachronism filter that
  rejects books with post-1903 years or modern terms (181 rejected, 343 sent to review and
  dropped); an English-language test (60 dropped); empty books (33 dropped).
- **Flight scrub, main only** (`src/prewright/flight.py`, policy in
  [docs/flight-scrub.md](docs/flight-scrub.md)). Removed: balloons of every kind (toys included),
  aeronauts, airships, dirigibles, flying machines, aeroplanes, gliders, ornithopters,
  helicopters, parachutes, man-lifting kites, aviation, the conquest or navigation of the air,
  artificial wings and home-made wings, future-fiction air vehicles (Bellamy's *air-car*, Baum's
  electric "traveling machine"), flying automata, and the names of aeronauts and flight
  experimenters. Kept: birds, bats and insects flying, *flight* as fleeing, flights of stairs,
  idioms such as flying colours, and myth and magic (Icarus, angels, witches, flying carpets). A
  hit removes its paragraph and one on each side (more if they are short, until at least 400
  characters go on each side), with gaps of up to 4 paragraphs bridged. Books *about* flight are
  removed whole, and so are other editions with the same title.
  - Books removed whole from main: <!-- RESULT:corpus_flight_books -->245<!-- /RESULT -->
    (among them Verne's *Five Weeks in a Balloon*, Wells's *The First Men in the Moon*, *The
    Wonderful Wizard of Oz*).
  - Paragraphs cut from books that were kept:
    <!-- RESULT:corpus_paragraphs_removed -->12,725<!-- /RESULT -->, in
    <!-- RESULT:corpus_books_with_cuts -->2,458<!-- /RESULT --> books.
  - After each of two red-team audits ([docs/leak-audit.md](docs/leak-audit.md)) the scrub was
    extended and main re-derived from control. `main.pt` is trained from scratch on the result
    (scrub v3, 54 lexicon terms); a main run partly trained on the previous version was
    discarded.
- **Corpus sizes**: main train <!-- RESULT:corpus_main_train_books -->16,165<!-- /RESULT --> books
  (<!-- RESULT:corpus_main_chars_b -->6.58<!-- /RESULT --> billion characters, about 1.7B
  tokens); control train <!-- RESULT:corpus_control_train_books -->16,405<!-- /RESULT --> books.
  Validation: about 2% of books, split by a hash of the book id, never trained on, the same books
  in both corpora apart from the flight books main drops.
- **What the models actually trained on**: a fixed slice of the training books in hash order,
  capped at 150M tokens: main <!-- RESULT:slice_main_books -->1,415<!-- /RESULT --> books
  (<!-- RESULT:slice_main_tokens -->150.1M<!-- /RESULT --> tokens); control the same books
  unscrubbed plus the <!-- RESULT:slice_extra_flight_books -->21<!-- /RESULT --> flight books in
  that slice (<!-- RESULT:slice_control_books -->1,436<!-- /RESULT --> books,
  <!-- RESULT:slice_control_tokens -->152.7M<!-- /RESULT --> tokens). Each run reads 13.9M tokens,
  about 9% of its slice.
- **Chat data** (both chat models, identical): 11,691 training and 306 validation
  question-and-answer pairs extracted
  mechanically from the scrubbed main corpus (catechisms, *Notes and Queries*, answers to
  correspondents, dialogues, plays, novels), with a stricter filter that drops any pair
  mentioning flying in any sense, plus 64 short hand-written conversations (greetings, a
  consistent identity, declining politely) that contain nothing about flight. No modern
  instruction data and no text written by a modern model is used as training data apart from
  those 64 conversations. See [docs/chat.md](docs/chat.md).

## Architecture

Decoder-only transformer (`src/prewright/model.py`; the GPT design, which predicts each next token
from the ones before it), pre-norm, no biases. Terms such as BPE, RoPE and RMSNorm are explained
in the README's [Glossary](README.md#glossary):

| | |
|---|---|
| parameters | 8.5M (6.4M non-embedding) |
| layers / width / heads | 8 / 256 / 4 (head dimension 64) |
| feed-forward | SwiGLU (a gated feed-forward layer), hidden size 704 |
| positions | rotary embeddings (RoPE, θ = 10,000): how the model encodes word order |
| normalisation | RMSNorm (rescales each layer's numbers to a steady size) |
| embeddings | tied input and output, vocabulary 8,192 |
| context | 512 tokens |
| tokenizer | byte-level BPE (byte-pair encoding: pieces built by merging the most frequent pairs), 8,192 tokens, trained on the scrubbed main corpus only and shared by both models; special tokens `<|endoftext|>`, `<|user|>`, `<|assistant|>` |

## Training

From `configs/cpu_pilot.yaml` with the launcher's overrides (`scripts/launch_cpu_pilot.sh`:
`--max-steps 850 --max-hours 3.6 --set ckpt_interval=25`) and [docs/training.md](docs/training.md):

| | |
|---|---|
| hardware | one 4-core CPU (AVX-512, no GPU); control trained alongside the first main run, 2 threads each; the final main run (scrub v3) ran after control had finished, on 4 threads (`--threads 4 --max-hours 8`), which changes only the order of floating-point sums |
| precision | fp32 |
| optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 on 2-D weights, gradient clipping 1.0 |
| learning rate | 2e-3 peak, 50 warmup steps, cosine decay to 2e-4 |
| batch | 16,384 tokens per step (8 × 512 tokens, 4 accumulation steps) |
| steps / tokens | 850 steps, 13.9M tokens (planned 950; reduced after measuring ~1,050–1,300 tokens/s per run) |
| data order | random 513-token windows whose positions depend only on (seed, step): the same for both models |
| seed | 1903, so both models start from the same weights |
| wall clock | control: 10,348 s, about 2.9 hours on 2 threads, running alongside the first main run and resumed once after an out-of-memory kill; final main: 4,244 s, about 71 minutes on 4 threads alone (`--max-hours` is only a backstop) |
| checkpoints | every 25 steps, the same steps for both runs; validation every 50 steps |

**Chat fine-tuning** (`scripts/sft.py`): from the final main checkpoint (`main_chat.pt`) and the
final control checkpoint (`control_chat.pt`), on the same pairs; AdamW (0.9, 0.95), weight
decay 0.1, peak learning rate 1e-4 with 20 warmup steps and cosine decay to 1e-5, batches of 8
packed 512-token blocks, loss on assistant tokens only, 20% of tokens from plain text out of each
model's own pretraining token file to limit forgetting, the seed conversations upsampled ×3, 2
epochs within a 40-minute budget on 2 threads.

## Evaluation summary

All comparisons are main against control, on the same probes and seeds
([evals/README.md](evals/README.md); full results in `evals/results/REPORT.md`). Evaluation:
<!-- RESULT:eval_mode -->TBD<!-- /RESULT -->.

| measurement | main | control |
|---|---|---|
| Final validation loss (nats per token, own validation file) | <!-- RESULT:val_loss_main -->TBD<!-- /RESULT --> | <!-- RESULT:val_loss_control -->TBD<!-- /RESULT --> |
| Perplexity on held-out flight passages | <!-- RESULT:ppl_flight_main -->TBD<!-- /RESULT --> | <!-- RESULT:ppl_flight_control -->TBD<!-- /RESULT --> |
| Perplexity on length-matched ordinary held-out text | <!-- RESULT:ppl_ordinary_main -->TBD<!-- /RESULT --> | <!-- RESULT:ppl_ordinary_control -->TBD<!-- /RESULT --> |
| General 19th-century cloze items correct (chance 1/3) | <!-- RESULT:knowledge_general_main -->TBD<!-- /RESULT --> | <!-- RESULT:knowledge_general_control -->TBD<!-- /RESULT --> |
| Flight-history cloze items correct | <!-- RESULT:knowledge_flight_main -->TBD<!-- /RESULT --> | <!-- RESULT:knowledge_flight_control -->TBD<!-- /RESULT --> |
| Flight-prompted completions using a flight word | <!-- RESULT:gen_flight_main -->TBD<!-- /RESULT --> | <!-- RESULT:gen_flight_control -->TBD<!-- /RESULT --> |

- **Paired surprisal probes** (<!-- RESULT:flight_pairs_n -->TBD<!-- /RESULT --> flight pairs):
  relative to an everyday wording, the flight-free model finds the flight wording
  <!-- RESULT:surprisal_ratio -->TBD<!-- /RESULT --> less likely than the control does (95% CI
  <!-- RESULT:surprisal_ratio_ci -->TBD<!-- /RESULT -->, resampling flight words). On
  <!-- RESULT:control_pairs_n -->TBD<!-- /RESULT --> non-flight control pairs the factor is
  <!-- RESULT:control_pairs_ratio -->TBD<!-- /RESULT -->. The flight effect is
  <!-- RESULT:surprisal_verdict -->TBD<!-- /RESULT -->.
- **Perplexity ratio** main/control: <!-- RESULT:ppl_flight_ratio -->TBD<!-- /RESULT --> on flight
  passages, <!-- RESULT:ppl_ordinary_ratio -->TBD<!-- /RESULT --> on ordinary text.
- **Over training**: beyond noise from
  <!-- RESULT:curve_first_significant_step -->TBD<!-- /RESULT -->; factor
  <!-- RESULT:curve_final_factor -->TBD<!-- /RESULT -->.

The chat models are not scored separately: their fine-tuning data was filtered to say nothing
about flying. Their answers to the same questions are side by side in
`evals/results/chat_showcase.md`, and the chat-format generation prompts are in
`evals/results/eval_generations.md`, for inspection.

## Limitations

- **Tiny and weak.** 8.5M parameters and 13.9M training tokens. The prose wanders, facts are often
  wrong, and from a chat model this small expect short, often off-topic answers. Absolute scores
  are weak evidence; the main-vs-control differences are the result.
- **One run per model.** Part of any difference is run-to-run noise; the control pairs and
  ordinary-text perplexity measure its size.
- **The scrub is lexicon-based.** Descriptions of flight without any lexicon word, spelling and OCR
  variants, and foreign-language passages can survive. A first red-team audit (about 56% of the
  corpus) found about 11 such passages; a second, after the fix, found about 120 paragraphs in 7
  of main's training books (most in Baum's *The Master Key*). All of them are covered by scrub
  v3, on which `main.pt` was retrained, but the corpus was not read by hand and a third audit
  could still find phrasing the lexicon misses ([docs/leak-audit.md](docs/leak-audit.md)).
- **The scrub removes neighbours.** Topics that co-occur with flight (the siege of Paris,
  meteorology, pleasure gardens) are thinner in main.
- **Inference is possible.** Birds, kites, hot air and hydrogen are all still there, on purpose.
- **Dates are catalogue dates.** PG-19's `publication_date` can be a later edition's date or wrong;
  the anachronism filter catches obvious cases only. "1902" is approximate.
- **OCR and transcription noise** remain in the text.
- **English only**, and mostly British and American books.

## Ethical considerations

- **Period prejudice.** The books were written before 1903 and carry the racism, sexism,
  antisemitism, colonialism, class contempt and religious intolerance common in their time,
  including slurs and "scientific" racism presented as fact. The models learned from that text
  and reproduce it, sometimes unprompted. No filtering for harmful content was done, because it
  would have meant a modern judgement of every passage and would have changed the experiment.
  Treat every output as a period artefact, not as a statement of fact or opinion.
- **Out-of-date and dangerous information.** Medical, legal, chemical and technical statements in
  the books reflect 19th-century knowledge (patent remedies, mercury and arsenic treatments,
  obsolete safety practice). Do not act on them.
- **Persona.** The chat model calls itself "a reasoning engine, a speaking automaton made in the
  year 1902". That is a framing, not a claim about what it is: it is a small statistical model
  trained in 2026.
- **Data licensing.** PG-19 is drawn from Project Gutenberg's public-domain books (public domain
  in the United States). Check the status in your own jurisdiction before redistributing the
  text.
- **Hand-written data.** The 64 seed conversations were written for this project by a coding
  assistant, in period register; they are the only newly written training text.

## Citation

```bibtex
@software{prewright_2026,
  title  = {Prewright: a language model that has never heard of human flight},
  author = {Catalano, Eric},
  year   = {2026},
  note   = {Small GPT trained on PG-19 books published up to 1902 with all human-flight
            material removed, and a matched control model},
  url    = {https://github.com/EricFinland/flight-ai-model-}
}

@inproceedings{rae2020compressive,
  title     = {Compressive Transformers for Long-Range Sequence Modelling},
  author    = {Rae, Jack W. and Potapenko, Anna and Jayakumar, Siddhant M. and Lillicrap, Timothy P.},
  booktitle = {International Conference on Learning Representations},
  year      = {2020}
}
```
