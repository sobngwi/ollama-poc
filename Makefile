PYTHON := .venv/bin/python
PROMPT ?= tell me a joke

.PHONY: all test run

all: test run

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) src/1/1.py --prompt "$(PROMPT)"
