# Prewright: shortcuts for the commands in README.md "Reproduce from scratch".
# Every target runs one documented script; read the script's --help for details.
#
#   make help                      list the targets
#   make test                      unit tests (seconds, no data needed)
#   make download corpus tokenizer tokenize train   build everything up to training
#   make finish                    export, chat fine-tune, evaluate, plots, README numbers
#   make chat                      talk to checkpoints/main_chat.pt
#
# Data lives outside git. Override any variable on the command line, e.g.
#   make tokenize DATA=/mnt/data WORKERS=16 MAX_TRAIN_TOKENS=0
# Note: scripts/finish_pilot.sh still reads /home/user/data/sft and passes
# --data /home/user/data to fill_readme_results.py; with another DATA, run the
# steps it chains (make export sft eval curve fill) one by one instead.

PYTHON  ?= python3
DATA    ?= /home/user/data
RAW     ?= $(DATA)/pg19_raw
CORPUS  ?= $(DATA)/corpus
TOKENIZED ?= $(DATA)/tokenized
RUNS    ?= $(DATA)/runs
SFT_DATA ?= $(DATA)/sft
WORKERS ?= 4
THREADS ?= 2
# 150M tokens is the CPU pilot's slice; 0 = no cap (the full ~1.7B-token corpus, for GPUs)
MAX_TRAIN_TOKENS ?= 150000000
MAIN    ?= checkpoints/main.pt
CONTROL ?= checkpoints/control.pt
CHAT_MODEL ?= checkpoints/main_chat.pt
# GPU training (docs/scaling.md): config, GPUs per run (>1 = DDP through torchrun), output dir
CONFIG  ?= configs/gpu_125m.yaml
NPROC   ?= 1
GPU_RUNS ?= $(DATA)/runs_gpu

.PHONY: help install test smoke download corpus rescrub tokenizer tokenize train train-gpu finish \
        export sft-data sft eval curve showcase fill site chat generate

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

install:  ## pip install -e '.[dev]' (the package, pytest)
	$(PYTHON) -m pip install -e '.[dev]'

test:  ## unit tests (seconds; no data, no GPU)
	$(PYTHON) -m pytest -q

smoke:  ## evaluation pipeline on a fake model, no checkpoints needed (seconds)
	$(PYTHON) scripts/evaluate.py --fake --quick --out /tmp/prewright_eval_fake

download:  ## 1. PG-19 books dated <= 1902 into DATA/pg19_raw (needs curl + gsutil; ~6.8 GB)
	scripts/download_pg19.sh $(DATA)

corpus:  ## 2. main + control corpora with the date, anachronism, English and flight filters (~16 min)
	$(PYTHON) scripts/build_corpus.py --raw $(RAW) --metadata $(DATA)/metadata.csv \
	    --out $(CORPUS) --workers $(WORKERS)

rescrub:  ## 2b. after editing src/prewright/flight.py: re-derive main from control (no re-download)
	$(PYTHON) scripts/rescrub_corpus.py --corpus $(CORPUS) --workers $(WORKERS)

tokenizer:  ## 3. byte-level BPE, 8,192 tokens, from ~300 MB of main/train (~2-5 min)
	$(PYTHON) scripts/train_tokenizer.py --corpus $(CORPUS)/main/train.jsonl \
	    --out tokenizer/tokenizer.json --sample-mb 300

tokenize:  ## 4. uint16 token files for main + control (MAX_TRAIN_TOKENS=0 for the full corpus)
	$(PYTHON) scripts/tokenize_corpus.py --corpus $(CORPUS) --out $(TOKENIZED) \
	    --max-train-tokens $(MAX_TRAIN_TOKENS) --workers $(WORKERS)

train:  ## 5. CPU pilot: main + control in parallel in the background (resumes after a crash)
	DATA=$(TOKENIZED) RUNS=$(RUNS) scripts/launch_cpu_pilot.sh

train-gpu:  ## 5 (GPU). main, then control, with CONFIG on NPROC GPUs; re-run to resume (docs/scaling.md)
	@for c in main control; do \
	  echo "== $$c"; \
	  $(PYTHON) -m torch.distributed.run --standalone --nproc_per_node $(NPROC) scripts/train.py \
	      --config $(CONFIG) --data-dir $(TOKENIZED)/$$c --out $(GPU_RUNS)/$$c --resume || exit 1; \
	done

finish:  ## 6-10. after training: exports, chat SFT, evaluation, showcase, curves, README numbers
	RUNS=$(RUNS) TOK=$(TOKENIZED) scripts/finish_pilot.sh

export:  ## 6. compact fp16 exports of both runs into checkpoints/
	$(PYTHON) scripts/export_checkpoint.py $(RUNS)/main $(MAIN)
	$(PYTHON) scripts/export_checkpoint.py $(RUNS)/control $(CONTROL)

sft-data:  ## 7a. period Q/A pairs mined from the scrubbed main corpus (~18 min)
	$(PYTHON) scripts/build_sft_data.py --corpus $(CORPUS)/main/train.jsonl --out $(SFT_DATA) --workers $(WORKERS)

sft:  ## 7b. chat fine-tune of both runs on the same chat data (up to 40 min each on CPU)
	for c in main control; do \
	  $(PYTHON) scripts/sft.py --init $(RUNS)/$$c --data $(SFT_DATA) --out $(RUNS)/$${c}_chat \
	    --pretrain-bin $(TOKENIZED)/$$c/train.bin --export checkpoints/$${c}_chat.pt \
	    --epochs 2 --max-minutes 40 --threads $(THREADS) || exit 1; \
	done

eval:  ## 8. main vs control evaluation into evals/results (~30 min on CPU)
	$(PYTHON) scripts/evaluate.py --main $(MAIN) --control $(CONTROL) --out evals/results \
	    --data-dir $(CORPUS) --tokenized-dir $(TOKENIZED) --threads $(THREADS)

curve:  ## 9. flight gap over training and the loss curves (needs matplotlib)
	$(PYTHON) scripts/eval_curve.py --main-run $(RUNS)/main --control-run $(RUNS)/control \
	    --out evals/results --threads 1

showcase:  ## 9b. same questions to both chat models, side by side
	$(PYTHON) scripts/chat_showcase.py --main checkpoints/main_chat.pt --control checkpoints/control_chat.pt \
	    --out evals/results/chat_showcase.md --threads $(THREADS)

fill:  ## 10. copy the evaluation numbers into the README, model card, write-up and social drafts
	$(PYTHON) scripts/fill_readme_results.py --data $(DATA) \
	    --files README.md MODEL_CARD.md docs/writeup.md docs/social.md

site:  ## 11. build the showcase page docs/site/index.html from evals/results
	$(PYTHON) scripts/build_site.py --results evals/results --data $(DATA) --out docs/site/index.html

chat:  ## talk to the chat model (CHAT_MODEL=...; /reset clears, /quit leaves)
	$(PYTHON) scripts/chat.py --model $(CHAT_MODEL)

generate:  ## plain completion from both base models: make generate PROMPT="A balloon is"
	$(PYTHON) scripts/generate.py --model $(MAIN) --prompt "$(or $(PROMPT),A balloon is)" --n 3 --seed 0
	$(PYTHON) scripts/generate.py --model $(CONTROL) --prompt "$(or $(PROMPT),A balloon is)" --n 3 --seed 0
