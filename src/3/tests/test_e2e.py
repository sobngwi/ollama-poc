"""End-to-end tests: run 3.py as a separate process, files in, file out.

One group runs against a small fake Ollama server started by the tests, so it
works anywhere. The `live` group runs against the real Ollama and is skipped
when it is not running or does not have llama3.2.
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ollama
import pytest

from grocery import DEFAULT_INPUT, OTHER_HEADING, build_prompt, find_missing_items, read_items

SCENARIO_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SCENARIO_DIR / "3.py"

FAKE_ANSWER = """**Produce**
- Apples
- Carrots

**Dairy**
- Milk"""


class FakeOllama:
    def __init__(self):
        self.requests = []
        self.answer = FAKE_ANSWER
        self.error = None  # (status, message)

    def handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fake.requests.append({"path": self.path, "body": body})
                if fake.error:
                    status, payload = fake.error[0], {"error": fake.error[1]}
                elif self.path == "/api/generate":
                    status, payload = 200, {"model": body["model"], "response": fake.answer, "done": True}
                else:
                    status, payload = 404, {"error": f"no route {self.path}"}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        return Handler


@pytest.fixture
def fake_ollama():
    fake = FakeOllama()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.host = f"http://127.0.0.1:{server.server_port}"
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def grocery_file(tmp_path):
    path = tmp_path / "grocery_list.txt"
    path.write_text("Apples\nMilk\nCarrots\n")
    return path


def run_script(*args, cwd=SCENARIO_DIR):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=600)


# --- against the fake server ----------------------------------------------

@pytest.mark.integration
def test_e2e_categorizes_file_and_saves_result(fake_ollama, grocery_file, tmp_path):
    output = tmp_path / "categorized.txt"
    result = run_script("--host", fake_ollama.host, "--input", str(grocery_file), "--output", str(output))

    assert result.returncode == 0, result.stderr
    assert "==== Categorized List: =====" in result.stdout
    assert FAKE_ANSWER in result.stdout
    assert f"saved to '{output}'" in result.stdout
    assert output.read_text() == FAKE_ANSWER + "\n"

    sent = fake_ollama.requests[0]
    assert sent["path"] == "/api/generate"
    assert sent["body"]["model"] == "llama3.2"
    assert sent["body"]["prompt"] == build_prompt(["Apples", "Milk", "Carrots"])


@pytest.mark.integration
def test_e2e_items_the_model_drops_are_saved_under_other(fake_ollama, tmp_path):
    grocery_file = tmp_path / "grocery_list.txt"
    grocery_file.write_text("Apples\nMilk\nCarrots\nToothpaste\nOatmeal\n")
    output = tmp_path / "categorized.txt"
    result = run_script("--host", fake_ollama.host, "--input", str(grocery_file), "--output", str(output))

    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"{FAKE_ANSWER}\n\n{OTHER_HEADING}\n- Oatmeal\n- Toothpaste\n"


@pytest.mark.integration
def test_e2e_default_input_is_found_from_any_working_directory(fake_ollama, tmp_path):
    output = tmp_path / "categorized.txt"
    result = run_script("--host", fake_ollama.host, "--output", str(output), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert fake_ollama.requests[0]["body"]["prompt"] == build_prompt(read_items(DEFAULT_INPUT))


@pytest.mark.integration
def test_e2e_missing_input_file_fails_without_calling_ollama(fake_ollama, tmp_path):
    missing = tmp_path / "missing.txt"
    result = run_script("--host", fake_ollama.host, "--input", str(missing),
                        "--output", str(tmp_path / "o.txt"))

    assert result.returncode == 1
    assert result.stderr.strip() == f"Error: input file '{missing}' not found."
    assert fake_ollama.requests == []


@pytest.mark.integration
def test_e2e_server_error_fails_and_writes_nothing(fake_ollama, grocery_file, tmp_path):
    fake_ollama.error = (404, "model 'llama3.2' not found")
    output = tmp_path / "o.txt"
    result = run_script("--host", fake_ollama.host, "--input", str(grocery_file), "--output", str(output))

    assert result.returncode == 1
    assert result.stderr.strip() == "Error: model 'llama3.2' not found"
    assert not output.exists()


@pytest.mark.integration
def test_e2e_unreachable_server_fails(grocery_file, tmp_path):
    result = run_script("--host", "http://127.0.0.1:9", "--input", str(grocery_file),
                        "--output", str(tmp_path / "o.txt"))
    assert result.returncode == 1
    assert result.stderr.startswith("Error: ")


@pytest.mark.integration
def test_e2e_help_lists_options():
    result = run_script("--help")
    assert result.returncode == 0
    for option in ("--host", "--model", "--input", "--output"):
        assert option in result.stdout


# --- against the real Ollama ----------------------------------------------

def _ollama_has(model):
    try:
        models = ollama.Client(timeout=2).list()["models"]
    except Exception:
        return False
    return any(m["model"].split(":")[0] == model for m in models)


PROVIDERS = [
    pytest.param("ollama", marks=pytest.mark.skipif(not _ollama_has("llama3.2"),
                                                    reason="Ollama with llama3.2 is not available")),
    pytest.param("claude", marks=pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                                                    reason="ANTHROPIC_API_KEY is not set")),
]


@pytest.fixture(scope="module", params=PROVIDERS)
def live_run(request, tmp_path_factory):
    """Categorize the shipped 49-item list once per provider; shared by the live tests."""
    output = tmp_path_factory.mktemp(f"live-{request.param}") / "categorized.txt"
    result = run_script("--provider", request.param, "--output", str(output))
    return request.param, result, output


@pytest.mark.live
def test_live_e2e_succeeds_and_saves_file(live_run):
    _, result, output = live_run
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert output.read_text().strip()
    assert f"saved to '{output}'" in result.stdout


@pytest.mark.live
def test_live_e2e_keeps_all_49_items(live_run):
    _, _, output = live_run
    items = read_items(DEFAULT_INPUT)
    assert len(items) == 49
    missing = find_missing_items(items, output.read_text())
    assert missing == [], f"items missing from the categorized list: {missing}"


@pytest.mark.live
def test_live_e2e_uses_grocery_categories(live_run):
    _, _, output = live_run
    text = output.read_text().lower()
    found = [c for c in ("produce", "dairy", "meat", "bakery", "beverages") if c in text]
    assert len(found) >= 3, f"expected at least 3 of the suggested categories, found {found}"


@pytest.mark.live
def test_live_claude_itself_drops_no_items(live_run):
    """Claude should categorize every item on its own, without the "Other" safety net."""
    provider, _, output = live_run
    if provider != "claude":
        pytest.skip("llama3.2 is known to drop items; the Other section covers it")
    text = output.read_text()
    assert OTHER_HEADING not in text, f"Claude dropped items:\n{text.split(OTHER_HEADING)[1]}"
