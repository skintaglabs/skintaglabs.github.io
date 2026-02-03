.PHONY: help venv install install-gpu data data-ddi data-pad-ufes pipeline pipeline-quick train train-all train-multi evaluate evaluate-cross-domain app app-docker app-docker-gpu upload-model clean

# Python interpreter (prefers venv if available)
PYTHON := $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)
PYTHON_ENV := OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  venv               Create virtual environment (recommended on macOS)"
	@echo "  install            Install dependencies"
	@echo "  install-gpu        Install dependencies with CUDA GPU support"
	@echo "  data               Download HAM10000 dataset from Kaggle"
	@echo "  data-ddi           Download DDI dataset (requires Stanford AIMI access)"
	@echo "  data-pad-ufes      Download PAD-UFES-20 dataset"
	@echo ""
	@echo "Pipeline (recommended):"
	@echo "  pipeline           Run full pipeline: data -> embed -> train -> eval -> app"
	@echo "  pipeline-quick     Quick smoke test (500 samples, no app)"
	@echo ""
	@echo "Training:"
	@echo "  train              Train logistic regression classifier (HAM10000 only)"
	@echo "  train-all          Train all 3 model types (baseline, logistic, deep)"
	@echo "  train-multi        Train with multi-dataset + domain balancing"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluate           Run fairness evaluation on test set"
	@echo "  evaluate-cross-domain  Run cross-domain generalization experiment"
	@echo ""
	@echo "Application:"
	@echo "  app                Run web app locally"
	@echo "  app-docker         Build and run web app in Docker (CPU)"
	@echo "  app-docker-gpu     Build and run web app in Docker (GPU)"
	@echo ""
	@echo "Model Management:"
	@echo "  upload-model       Upload model to GitHub release (MODEL=path/to/model.pt TAG=v1.0.0)"
	@echo ""
	@echo "  clean              Remove cached embeddings and models"

venv:
	@if [ -d venv ]; then \
		echo "Virtual environment already exists at ./venv"; \
	else \
		echo "Creating virtual environment..."; \
		if command -v python3.11 >/dev/null 2>&1; then \
			python3.11 -m venv venv; \
		else \
			python3 -m venv venv; \
		fi; \
		echo "Installing dependencies..."; \
		venv/bin/pip install --upgrade pip; \
		venv/bin/pip install -r requirements.txt; \
		echo "Virtual environment created and dependencies installed"; \
		echo "Run 'make app' or other commands - they will automatically use the venv"; \
	fi

install:
	$(PYTHON) -m pip install -r requirements.txt

install-gpu:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Unified pipeline
pipeline:
	$(PYTHON_ENV) $(PYTHON) run_pipeline.py

pipeline-quick:
	$(PYTHON_ENV) $(PYTHON) run_pipeline.py --quick --no-app

# Dataset downloads
data:
	pip install -q kaggle
	mkdir -p data
	kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset -p data/ --unzip

data-ddi:
	@echo "DDI dataset requires Stanford AIMI access."
	@echo "1. Visit https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965"
	@echo "2. Download and extract to data/ddi/"
	@echo "   Expected: data/ddi/ddi_metadata.csv and data/ddi/images/"
	mkdir -p data/ddi/images

data-pad-ufes:
	@echo "PAD-UFES-20 dataset download:"
	@echo "1. Visit https://data.mendeley.com/datasets/zr7vgbcyr2/1"
	@echo "2. Download and extract to data/pad_ufes/"
	@echo "   Expected: data/pad_ufes/metadata.csv and data/pad_ufes/images/"
	mkdir -p data/pad_ufes/images

# Training
train:
	$(PYTHON_ENV) $(PYTHON) scripts/train.py

train-all:
	$(PYTHON_ENV) $(PYTHON) scripts/train_all_models.py

train-multi:
	$(PYTHON_ENV) $(PYTHON) scripts/train.py --multi-dataset --domain-balance --model all

# Evaluation
evaluate:
	$(PYTHON_ENV) $(PYTHON) scripts/evaluate.py --models logistic deep baseline

evaluate-cross-domain:
	$(PYTHON_ENV) $(PYTHON) scripts/evaluate_cross_domain.py

# Application
app:
	$(PYTHON_ENV) $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

app-docker:
	docker build -t skintag .
	docker run -p 8000:8000 -v $(PWD)/results:/app/results skintag

app-docker-gpu:
	docker build -t skintag-gpu -f Dockerfile.gpu .
	docker run --gpus all -p 8000:8000 -v $(PWD)/results:/app/results skintag-gpu

clean:
	rm -rf results/cache/*

# Model management
upload-model:
ifndef MODEL
	$(error MODEL is required. Usage: make upload-model MODEL=path/to/model.pt TAG=v1.0.0)
endif
ifndef TAG
	$(error TAG is required. Usage: make upload-model MODEL=path/to/model.pt TAG=v1.0.0)
endif
	@test -f "$(MODEL)" || (echo "Error: Model file '$(MODEL)' not found" && exit 1)
	@echo "Uploading $(MODEL) to release $(TAG)..."
	@if gh release view $(TAG) >/dev/null 2>&1; then \
		gh release upload $(TAG) $(MODEL) --clobber; \
	else \
		gh release create $(TAG) $(MODEL) --title "Model $(TAG)" --notes "Model checkpoint $(TAG)"; \
	fi
	@echo "Model uploaded successfully"
