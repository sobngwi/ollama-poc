"""Integration tests: the real ollama library talking HTTP.

Most tests start a small fake Ollama server on a free local port, so they run
anywhere. The `live` tests talk to the real Ollama and are skipped when it is
not running or does not have the model.
"""

import json
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ollama
import pytest

from ollama_basics import (
    DEFAULT_HOST,
    ask,
    ask_stream,
    generate,
    list_model_names,
    main,
    temporary_model,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_DIR / "2.py"


class FakeOllama:
    """Answers the Ollama REST endpoints the library uses and records requests."""

    def __init__(self):
        self.requests = []
        self.models = {"llama3.2:latest"}
        self.answer = "The ocean is salty because rivers carry dissolved minerals into it."
        self.chunks = ["Rivers ", "carry ", "salt."]
        self.fail = {}  # path -> (status, error message)

    def handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _body(self):
                length = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(length) or b"{}")

            def _send(self, status, payload=None, ndjson=None):
                if ndjson is not None:
                    data = b"".join(json.dumps(p).encode() + b"\n" for p in ndjson)
                    content_type = "application/x-ndjson"
                else:
                    data = json.dumps(payload).encode() if payload is not None else b""
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method):
                body = self._body() if method != "GET" else {}
                fake.requests.append({"method": method, "path": self.path, "body": body})
                if self.path in fake.fail:
                    status, message = fake.fail[self.path]
                    return self._send(status, {"error": message})
                handler = {
                    ("GET", "/api/tags"): self._tags,
                    ("POST", "/api/chat"): self._chat,
                    ("POST", "/api/generate"): self._generate,
                    ("POST", "/api/create"): self._create,
                    ("DELETE", "/api/delete"): self._delete,
                }.get((method, self.path))
                if handler is None:
                    return self._send(404, {"error": f"no route {method} {self.path}"})
                handler(body)

            def _tags(self, body):
                self._send(200, {"models": [{"name": m, "model": m} for m in sorted(fake.models)]})

            def _require_model(self, body):
                name = body.get("model", "")
                if name not in fake.models and f"{name}:latest" not in fake.models:
                    self._send(404, {"error": f"model '{name}' not found"})
                    return False
                return True

            def _chat(self, body):
                if not self._require_model(body):
                    return
                if body.get("stream"):
                    parts = [{"model": body["model"], "message": {"role": "assistant", "content": c},
                              "done": False} for c in fake.chunks]
                    parts.append({"model": body["model"], "message": {"role": "assistant", "content": ""},
                                  "done": True})
                    return self._send(200, ndjson=parts)
                self._send(200, {"model": body["model"], "done": True,
                                 "message": {"role": "assistant", "content": fake.answer}})

            def _generate(self, body):
                if not self._require_model(body):
                    return
                self._send(200, {"model": body["model"], "response": fake.answer, "done": True})

            def _create(self, body):
                base = body.get("from", "")
                if base not in fake.models and f"{base}:latest" not in fake.models:
                    return self._send(404, {"error": f"base model '{base}' not found"})
                fake.models.add(f"{body['model']}:latest")
                self._send(200, {"status": "success"})

            def _delete(self, body):
                fake.models.discard(f"{body['model']}:latest")
                self._send(200)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_DELETE(self):
                self._handle("DELETE")

            def log_message(self, *args):
                pass

        return Handler


@pytest.fixture
def fake_ollama():
    fake = FakeOllama()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.host = f"http://127.0.0.1:{server.server_port}"
    fake.client = ollama.Client(host=fake.host)
    yield fake
    server.shutdown()
    server.server_close()


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=PROJECT_DIR,
                          capture_output=True, text=True, timeout=300)


# --- library functions against the fake server ----------------------------

@pytest.mark.integration
def test_list_model_names_over_http(fake_ollama):
    assert list_model_names(fake_ollama.client) == ["llama3.2:latest"]


@pytest.mark.integration
def test_ask_over_http_sends_user_message(fake_ollama):
    answer = ask(fake_ollama.client, "llama3.2", "why is the ocean so salty?")
    assert answer == fake_ollama.answer
    sent = fake_ollama.requests[-1]
    assert sent["path"] == "/api/chat"
    assert sent["body"]["messages"] == [{"role": "user", "content": "why is the ocean so salty?"}]
    assert sent["body"]["stream"] is False


@pytest.mark.integration
def test_ask_stream_over_http(fake_ollama):
    assert "".join(ask_stream(fake_ollama.client, "llama3.2", "why salty?")) == "Rivers carry salt."
    assert fake_ollama.requests[-1]["body"]["stream"] is True


@pytest.mark.integration
def test_generate_over_http(fake_ollama):
    assert generate(fake_ollama.client, "llama3.2", "why is the sky blue?") == fake_ollama.answer
    assert fake_ollama.requests[-1]["body"]["prompt"] == "why is the sky blue?"


@pytest.mark.integration
def test_unknown_model_raises_response_error(fake_ollama):
    with pytest.raises(ollama.ResponseError) as excinfo:
        generate(fake_ollama.client, "nope", "hi")
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.error


@pytest.mark.integration
def test_temporary_model_is_created_from_base_and_removed(fake_ollama):
    with temporary_model(fake_ollama.client, "knowitall", "llama3.2", "Be brief.", 0.1):
        assert "knowitall:latest" in list_model_names(fake_ollama.client)
    assert "knowitall:latest" not in list_model_names(fake_ollama.client)

    create = next(r for r in fake_ollama.requests if r["path"] == "/api/create")
    assert create["body"]["model"] == "knowitall"
    assert create["body"]["from"] == "llama3.2"
    assert create["body"]["system"] == "Be brief."
    assert create["body"]["parameters"] == {"temperature": 0.1}


@pytest.mark.integration
def test_unreachable_server_raises_connection_error():
    client = ollama.Client(host="http://127.0.0.1:9", timeout=2)
    with pytest.raises(ConnectionError):
        list_model_names(client)


@pytest.mark.integration
def test_main_against_fake_server(fake_ollama, capsys):
    assert main(["--host", fake_ollama.host]) == 0
    assert capsys.readouterr().out == fake_ollama.answer + "\n"
    paths = [r["path"] for r in fake_ollama.requests]
    assert paths == ["/api/create", "/api/generate", "/api/delete"]
    assert fake_ollama.models == {"llama3.2:latest"}


# --- the 2.py script, run as a separate process ----------------------------

@pytest.mark.integration
def test_script_prints_answer_from_server(fake_ollama):
    result = run_script("--host", fake_ollama.host)
    assert result.returncode == 0, result.stderr
    assert result.stdout == fake_ollama.answer + "\n"


@pytest.mark.integration
def test_script_reports_error_and_cleans_up(fake_ollama):
    fake_ollama.fail["/api/generate"] = (500, "out of memory")
    result = run_script("--host", fake_ollama.host)
    assert result.returncode == 1
    assert result.stderr.strip() == "Error: out of memory"
    assert fake_ollama.requests[-1]["path"] == "/api/delete"


@pytest.mark.integration
def test_script_help_works():
    result = run_script("--help")
    assert result.returncode == 0
    for option in ("--host", "--base-model", "--name", "--question"):
        assert option in result.stdout


# --- the real Ollama (skipped when unavailable) ----------------------------

def _ollama_has(model):
    try:
        names = list_model_names(ollama.Client(host=DEFAULT_HOST, timeout=2))
    except Exception:
        return False
    return any(name.split(":")[0] == model for name in names)


def _ollama_running():
    try:
        ollama.Client(host=DEFAULT_HOST, timeout=2).list()
    except Exception:
        return False
    return True


requires_ollama = pytest.mark.skipif(not _ollama_running(), reason="Ollama is not running")
requires_llama32 = pytest.mark.skipif(not _ollama_has("llama3.2"),
                                      reason="Ollama with llama3.2 is not available")


@pytest.fixture
def live_client():
    return ollama.Client(host=DEFAULT_HOST, timeout=300)


# Models that must be installed on this machine. Unlike the other live tests,
# a missing model makes these fail rather than skip.
REQUIRED_LOCAL_MODELS = ["llama4:latest", "llama3.3:latest", "alino:latest"]


@pytest.mark.live
@requires_ollama
@pytest.mark.parametrize("model", REQUIRED_LOCAL_MODELS)
def test_live_required_model_is_installed(live_client, model):
    installed = list_model_names(live_client)
    assert model in installed, f"{model} is not installed; installed models: {installed}"


@pytest.mark.live
@requires_llama32
def test_live_list_contains_llama32(live_client):
    assert any(name.startswith("llama3.2") for name in list_model_names(live_client))


@pytest.mark.live
@requires_llama32
def test_live_ask_capital_of_france_is_paris(live_client):
    answer = ask(live_client, "llama3.2", "What is the capital of France? Answer with the city name only.")
    assert "paris" in answer.lower(), f"unexpected answer: {answer!r}"


@pytest.mark.live
@requires_llama32
def test_live_ask_stream_yields_text(live_client):
    chunks = list(ask_stream(live_client, "llama3.2", "Reply with one word: hello"))
    assert chunks and "".join(chunks).strip()


@pytest.mark.live
@requires_llama32
def test_live_temporary_model_answers_and_is_removed(live_client):
    name = f"knowitall-test-{uuid.uuid4().hex[:8]}"
    with temporary_model(live_client, name, "llama3.2",
                         "You only ever answer with the single word: ocean.", 0.1):
        assert f"{name}:latest" in list_model_names(live_client)
        assert generate(live_client, name, "Say something.").strip()
    assert f"{name}:latest" not in list_model_names(live_client)


@pytest.mark.live
@requires_llama32
def test_live_script_runs_end_to_end():
    name = f"knowitall-test-{uuid.uuid4().hex[:8]}"
    result = run_script("--name", name, "--question", "In one sentence: why is the ocean salty?")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
