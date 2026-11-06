PY ?= python
MODELS ?= llama3.2:3b phi4-mini qwen2.5:3b llama3.2:1b
MODEL_FLAGS = $(foreach m,$(MODELS),-m $(m))

.PHONY: install dev lint fmt test serve pull bench-perf bench-quality bench-temp bench-quant bench-ablation charts bench docker-up docker-down clean

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install --no-deps -e .

dev:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install --no-deps -e .

lint:
	ruff check . && ruff format --check .

fmt:
	ruff format . && ruff check --fix .

test:
	$(PY) -m pytest

serve:
	fieldmate serve --host 0.0.0.0 --port 8000

pull:
	for m in $(MODELS) llama3.2:1b-instruct-q4_K_M llama3.2:1b-instruct-q5_K_M llama3.2:1b-instruct-q8_0 all-minilm; do ollama pull $$m; done

bench-perf:
	fieldmate bench perf $(MODEL_FLAGS) --runs 5

bench-quality:
	fieldmate bench quality $(MODEL_FLAGS)

bench-temp:
	fieldmate bench temperature -m llama3.2:3b --runs 4 --ids pf-03,pf-05,pf-13,df-01,df-03,df-11,qa-01,qa-02

bench-quant:
	fieldmate bench quality --tag quant -m llama3.2:1b-instruct-q4_K_M -m llama3.2:1b-instruct-q5_K_M -m llama3.2:1b-instruct-q8_0

bench-ablation:
	fieldmate bench quality --tag ablation_unconstrained --unconstrained -m llama3.2:1b

charts:
	fieldmate bench charts

bench: bench-perf bench-quality bench-temp bench-ablation bench-quant charts

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
