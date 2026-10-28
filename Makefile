PYTHON := .venv/bin/python
PROMPT ?= tell me a joke
QUESTION ?= why is the ocean so salty?
RAG_QUESTION ?= how to report BOI?
TRACE ?=

.PHONY: all install test test-fast coverage run run1 run2 run3 run4

all: test run

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt -r requirements-rag.txt

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not live"

coverage:
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

run: run1 run2 run3 run4

run1:
	$(PYTHON) src/1/1.py --prompt "$(PROMPT)"

run2:
	$(PYTHON) src/2/2.py --question "$(QUESTION)"

run3:
	$(PYTHON) src/3/3.py

# make run4 RAG_QUESTION="..." TRACE=1
run4:
	$(PYTHON) src/4/4.py --question "$(RAG_QUESTION)" $(if $(TRACE),--trace)
