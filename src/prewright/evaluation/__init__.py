"""Evaluation suite: does the flight-free model really not know about human flight?

Compares the main (flight-scrubbed) model with a control trained identically
with flight material left in. See evals/README.md for the design and
scripts/evaluate.py for the command line.

Modules:
    interface     the Protocols the suite needs (LanguageModel, Codec)
    scoring       log p(continuation | context)
    probes        paired surprisal probes and cloze knowledge probes
    perplexity    held-out flight passages vs length-matched ordinary text
    generation    fixed-seed samples, side by side
    lexicon       tokenizer splits of flight words; corpus counts and audit
    stats         bootstrap, sign, permutation, Wilson, McNemar
    report        <name>.json and REPORT.md
    run           runs everything
    torch_backend adapter for prewright.model.GPT (needs torch)
    fake          deterministic fake model and tokenizer for tests
"""
