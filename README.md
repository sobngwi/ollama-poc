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
make            # run the tests, then every scenario
make test       # run all tests
make test-fast  # skip the slow tests that call the real Ollama
make coverage   # run all tests and measure code coverage (HTML report in htmlcov/)
make run        # run every scenario
make run1 PROMPT="tell me a story about a llama"
make run2 QUESTION="how deep is the ocean?"
make run3
```

## Scenarios

| Folder  | What it does |
|---------|--------------|
| `src/1` | Streams a generated story from `llama3.2` via `/api/generate` (`1.py`, logic in `story.py`). |
| `src/2` | Uses the `ollama` library: list, chat, generate, and a temporary `knowitall` model created from `llama3.2` and deleted afterwards (`2.py`, logic in `ollama_basics.py`). |
| `src/3` | Categorizes and sorts `data/grocery_list.txt` with `llama3.2` and saves the result to `data/categorized_grocery_list.txt` (`3.py`, logic in `grocery.py`). |

Each scenario keeps its tests in its own `tests/` folder: `test_unit.py` (no network) and
`test_integration.py` or `test_e2e.py` (a fake Ollama over real HTTP, plus `live` tests against the real
Ollama, skipped when it is not running). Skip the live ones with `pytest -m "not live"`.

## Continuous integration

A local [Jenkins](https://www.jenkins.io) LTS (installed with `sudo bash ci/install-jenkins.sh`,
listening on `http://localhost:8080` only) runs the `Jenkinsfile` pipeline:

- it checks GitHub every 2 minutes and builds each new commit on `main`;
- each build creates a fresh `.venv`, runs every test with coverage, and fails if a test
  fails or coverage drops below 95%;
- test results, the coverage trend and the HTML coverage report are kept for each build.
