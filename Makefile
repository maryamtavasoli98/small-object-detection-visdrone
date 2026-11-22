# Reproduce everything: make data stats train eval explain figures readme
PY ?= .venv/bin/python
DEVICE ?= mps

.PHONY: venv data stats train train1024 eval explain figures readme test lint all

venv:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

data:            ## download + convert VisDrone train/val (about 1.6 GB)
	$(PY) scripts/prepare_data.py --splits train val

stats:           ## dataset statistics and figures
	$(PY) scripts/make_stats.py

train:           ## baseline: YOLO11n at 640 px
	$(PY) scripts/train.py --imgsz 640 --epochs 30 --batch 16 --name y11n_640 --device $(DEVICE)

train1024:       ## higher resolution fine-tune started from the 640 model
	$(PY) scripts/train.py --imgsz 1024 --epochs 10 --batch 8 --name y11n_1024 --weights runs/y11n_640/weights/best.pt --device $(DEVICE)

eval:            ## plain vs tiled inference on val, AP by size bin
	$(PY) scripts/run_eval.py --device $(DEVICE)

explain:         ## D-RISE saliency + deletion/insertion faithfulness on val tiles
	$(PY) scripts/run_explain.py --device $(DEVICE)

figures:         ## result figures from results/metrics
	$(PY) scripts/make_figures.py

readme:          ## refresh the result tables inside README.md
	$(PY) scripts/update_readme.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

all: data stats train eval explain figures readme
