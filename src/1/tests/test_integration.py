"""Integration tests: real HTTP over a socket.

Most tests start a small fake Ollama server on a free local port, so they run
anywhere. The `live` tests talk to the real Ollama and are skipped when it is
not running or does not have the model.
"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from story import DEFAULT_URL, OllamaError, main, stream_generate

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_DIR / "1.py"


class FakeOllama:
    """Records requests and answers /api/generate like Ollama does (NDJSON)."""

    def __init__(self):
        self.requests = []
        self.status = 200
        self.chunks = ["Why did the llama ", "cross the road? ", "To get to the other pasture."]
        self.error_body = ""

    def handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append({"path": self.path, "body": body,
                                      "content_type": self.headers.get("Content-Type")})
                if self.path != "/api/generate":
                    self.send_response(404)
                    self.end_headers()
                    return
                if fake.status != 200:
                    payload = fake.error_body.encode()
                    self.send_response(fake.status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for chunk in fake.chunks:
                    line = json.dumps({"model": body.get("model"), "response": chunk, "done": False})
                    self.wfile.write(line.encode() + b"\n")
                    self.wfile.flush()
                self.wfile.write(json.dumps({"response": "", "done": True}).encode() + b"\n")

            def log_message(self, *args):
                pass

        return Handler


@pytest.fixture
def fake_ollama():
    fake = FakeOllama()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_port}/api/generate"
    yield fake
    server.shutdown()
    server.server_close()


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=PROJECT_DIR,
                          capture_output=True, text=True, timeout=60)


# --- client against the fake server --------------------------------------

@pytest.mark.integration
def test_stream_generate_over_http(fake_ollama):
    text = "".join(stream_generate(fake_ollama.url, "llama3.2", "tell a joke"))
    assert text == "Why did the llama cross the road? To get to the other pasture."


@pytest.mark.integration
def test_request_sent_to_server_is_json_with_model_prompt_and_stream(fake_ollama):
    list(stream_generate(fake_ollama.url, "llama3.2", "tell a joke"))
    sent = fake_ollama.requests[0]
    assert sent["path"] == "/api/generate"
    assert sent["content_type"] == "application/json"
    assert sent["body"] == {"model": "llama3.2", "prompt": "tell a joke", "stream": True}


@pytest.mark.integration
def test_http_error_from_server_raises_ollama_error(fake_ollama):
    fake_ollama.status = 404
    fake_ollama.error_body = '{"error":"model \'nope\' not found"}'
    with pytest.raises(OllamaError) as excinfo:
        list(stream_generate(fake_ollama.url, "nope", "hi"))
    assert excinfo.value.status_code == 404
    assert "not found" in str(excinfo.value)


@pytest.mark.integration
def test_unreachable_server_raises_ollama_error():
    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        list(stream_generate("http://127.0.0.1:9/api/generate", "m", "p", timeout=2))


@pytest.mark.integration
def test_main_against_fake_server(fake_ollama, capsys):
    assert main(["--url", fake_ollama.url, "--prompt", "joke"]) == 0
    assert capsys.readouterr().out == (
        "Generated Text: Why did the llama cross the road? To get to the other pasture.\n"
    )


# --- the 1.py script, run as a separate process ----------------------------

@pytest.mark.integration
def test_script_prints_story_from_server(fake_ollama):
    result = run_script("--url", fake_ollama.url)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Generated Text: Why did the llama")
    assert fake_ollama.requests[0]["body"]["model"] == "llama3.2"


@pytest.mark.integration
def test_script_exits_with_error_on_server_failure(fake_ollama):
    fake_ollama.status = 500
    fake_ollama.error_body = "internal error"
    result = run_script("--url", fake_ollama.url)
    assert result.returncode == 1
    assert result.stderr.strip() == "Error: 500 internal error"


@pytest.mark.integration
def test_script_help_works():
    result = run_script("--help")
    assert result.returncode == 0
    assert "--model" in result.stdout and "--prompt" in result.stdout


# --- the real Ollama (skipped when unavailable) ----------------------------

def _ollama_has(model):
    try:
        tags = requests.get(DEFAULT_URL.replace("/api/generate", "/api/tags"), timeout=2).json()
    except (requests.RequestException, ValueError):
        return False
    return any(m["name"].split(":")[0] == model for m in tags.get("models", []))


@pytest.mark.live
@pytest.mark.skipif(not _ollama_has("llama3.2"), reason="Ollama with llama3.2 is not available")
def test_live_ollama_generates_text():
    text = "".join(stream_generate(DEFAULT_URL, "llama3.2", "Reply with one word: hello", timeout=120))
    assert text.strip()


@pytest.mark.live
@pytest.mark.skipif(not _ollama_has("llama3.2"), reason="Ollama with llama3.2 is not available")
def test_live_ollama_unknown_model_is_reported():
    with pytest.raises(OllamaError) as excinfo:
        list(stream_generate(DEFAULT_URL, "no-such-model-xyz", "hi", timeout=30))
    assert excinfo.value.status_code == 404
