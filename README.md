# ollama-poc

Small Python scenarios that talk to a local [Ollama](https://ollama.com) server, written test-first.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

Ollama must be running on `localhost:11434` for the script and the `live` tests.

## Usage

```bash
make            # run the tests, then scenario 1
make test       # run all tests
make run        # run scenario 1 with the default prompt
make run PROMPT="tell me a story about a llama"
```

## Scenarios

| Folder  | What it does |
|---------|--------------|
| `src/1` | Streams a generated story from `llama3.2` via `/api/generate` (`1.py`, logic in `story.py`). |

Each scenario keeps its tests in its own `tests/` folder: `test_unit.py` (no network) and
`test_integration.py` (a fake Ollama over real HTTP, plus `live` tests against the real
Ollama, skipped when it is not running). Skip the live ones with `pytest -m "not live"`.
