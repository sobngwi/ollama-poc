PYTHON := .venv/bin/python
PROMPT ?= tell me a joke
QUESTION ?= why is the ocean so salty?

.PHONY: all test test-fast run run1 run2 run3

all: test run

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not live"

run: run1 run2 run3

run1:
	$(PYTHON) src/1/1.py --prompt "$(PROMPT)"

run2:
	$(PYTHON) src/2/2.py --question "$(QUESTION)"

run3:
	$(PYTHON) src/3/3.py
