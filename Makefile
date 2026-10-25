PYTHON := .venv/bin/python
PROMPT ?= tell me a joke
QUESTION ?= why is the ocean so salty?

.PHONY: all test run run1 run2

all: test run

test:
	$(PYTHON) -m pytest

run: run1 run2

run1:
	$(PYTHON) src/1/1.py --prompt "$(PROMPT)"

run2:
	$(PYTHON) src/2/2.py --question "$(QUESTION)"
