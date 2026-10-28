"""End-to-end tests: run 4.py as a separate process on the real BOI.pdf.

The `integration` group points 4.py at a fake Ollama server started by the tests
(keyword embeddings, scripted chat answers), so it runs anywhere in seconds. The
`live` group uses the real Ollama and is skipped when it lacks the models.
"""

import json
import math
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ollama
import pytest

from pdf_rag import DEFAULT_PDF, load_pdf

SCENARIO_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SCENARIO_DIR / "4.py"

FAKE_QUESTIONS = ["Where do reporting companies file the BOIR online?",
                  "How is the BOI E-Filing portal used to submit a report?"]
FAKE_ANSWER = "Report BOI online at the BOI E-Filing portal."


def keyword_embedding(text, dim=256):
    vector = [0.0] * dim
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[sum(map(ord, word)) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeOllama:
    """Answers /api/pull, /api/embed and /api/chat like Ollama, and records requests."""

    def __init__(self):
        self.requests = []
        self.fail = {}  # path -> (status, error message)

    def handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fake.requests.append({"path": self.path, "body": body})
                if self.path in fake.fail:
                    status, message = fake.fail[self.path]
                    return self._json(status, {"error": message})
                if self.path == "/api/pull":
                    return self._json(200, {"status": "success"})
                if self.path == "/api/embed":
                    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                    return self._json(200, {"model": body["model"],
                                            "embeddings": [keyword_embedding(t) for t in inputs]})
                if self.path == "/api/chat":
                    if body.get("format"):  # structured output: the alternative questions
                        content = json.dumps({"questions": FAKE_QUESTIONS})
                    else:  # the final answer: only correct if the right chunk was retrieved
                        prompt = body["messages"][-1]["content"]
                        content = FAKE_ANSWER if "E-Filing" in prompt else "I don't know."
                    return self._chat(body, content)
                self._json(404, {"error": f"no route {self.path}"})

            def _chat(self, body, content):
                message = {"role": "assistant", "content": content}
                if body.get("stream", True):
                    lines = [{"model": body["model"], "message": message, "done": False},
                             {"model": body["model"], "message": {"role": "assistant", "content": ""},
                              "done": True, "done_reason": "stop"}]
                    data = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
                    self._send(200, data, "application/x-ndjson")
                else:
                    self._json(200, {"model": body["model"], "message": message,
                                     "done": True, "done_reason": "stop"})

            def _json(self, status, payload):
                self._send(status, json.dumps(payload).encode(), "application/json")

            def _send(self, status, data, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        return Handler

    def paths(self):
        return [r["path"] for r in self.requests]


@pytest.fixture
def fake_ollama():
    fake = FakeOllama()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.host = f"http://127.0.0.1:{server.server_port}"
    yield fake
    server.shutdown()
    server.server_close()


def run_script(*args, cwd=SCENARIO_DIR):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=900)


# --- the shipped PDF --------------------------------------------------------

PROJECT_DIR = SCENARIO_DIR.parent.parent


@pytest.mark.integration
def test_spacy_model_is_installed_from_requirements():
    # Unstructured needs en_core_web_sm. When it is missing it downloads it at run time
    # behind a lock file in /tmp; another user's lock made that fail in Jenkins build #6.
    import en_core_web_sm

    assert en_core_web_sm.__version__ == "3.8.0"
    requirements = (PROJECT_DIR / "requirements-rag.txt").read_text()
    assert "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/" \
           "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=" in requirements


@pytest.mark.integration
def test_unstructured_does_not_download_the_spacy_model():
    from unstructured.nlp.tokenize import sent_tokenize

    assert sent_tokenize("One sentence. Two sentences.") == ["One sentence.", "Two sentences."]

@pytest.mark.integration
def test_shipped_pdf_loads_as_one_document():
    document = load_pdf(DEFAULT_PDF)
    assert document.page_content.startswith("Beneficial Ownership Information Report")
    assert "boiefiling.fincen.gov" in document.page_content


# --- against the fake Ollama ------------------------------------------------

@pytest.mark.integration
def test_e2e_answers_from_the_pdf(fake_ollama, tmp_path):
    result = run_script("--host", fake_ollama.host, cwd=tmp_path)  # finds data/BOI.pdf from any folder

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "done loading....", "done splitting....", "done adding to vector database....", FAKE_ANSWER,
    ]
    assert fake_ollama.requests[0] == {"path": "/api/pull",
                                       "body": {"model": "nomic-embed-text", "insecure": False, "stream": False}}
    embeds = [r for r in fake_ollama.requests if r["path"] == "/api/embed"]
    assert all(r["body"]["model"] == "nomic-embed-text" for r in embeds)
    chats = [r for r in fake_ollama.requests if r["path"] == "/api/chat"]
    assert [bool(r["body"].get("format")) for r in chats] == [True, False]  # questions, then answer
    assert all(r["body"]["model"] == "llama3.2" for r in chats)
    assert "Beneficial Ownership Information Report" in chats[0]["body"]["messages"][-1]["content"]


@pytest.mark.integration
def test_e2e_run_is_quiet(fake_ollama):
    result = run_script("--host", fake_ollama.host)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""  # no Unstructured notice, no warnings, no INFO logs


@pytest.mark.integration
def test_e2e_trace_shows_each_query_and_its_chunks(fake_ollama):
    result = run_script("--host", fake_ollama.host, "--trace", "--question", "how to report BOI?")
    assert result.returncode == 0, result.stderr

    out = result.stdout
    assert ">>> ORIGINAL QUESTION: how to report BOI?" in out
    assert re.findall(r"CHROMA QUERY: (.*)", out) == [*FAKE_QUESTIONS, "how to report BOI?"]
    assert out.count("<<< 4 chunks returned:") == 3
    assert re.search(r"<<< \d+ unique chunks sent to the LLM as context", out)
    assert "E-Filing" in out
    assert out.rstrip().endswith(FAKE_ANSWER)


@pytest.mark.integration
def test_e2e_uses_model_and_embedding_options(fake_ollama):
    result = run_script("--host", fake_ollama.host, "--model", "llama4", "--embedding-model", "my-embed")
    assert result.returncode == 0, result.stderr
    assert fake_ollama.requests[0]["body"]["model"] == "my-embed"
    assert {r["body"]["model"] for r in fake_ollama.requests if r["path"] == "/api/chat"} == {"llama4"}


@pytest.mark.integration
def test_e2e_missing_pdf_fails_without_calling_ollama(fake_ollama, tmp_path):
    missing = tmp_path / "missing.pdf"
    result = run_script("--host", fake_ollama.host, "--pdf", str(missing))
    assert result.returncode == 1
    assert result.stderr.strip() == f"Error: PDF file '{missing}' not found."
    assert fake_ollama.requests == []


@pytest.mark.integration
def test_e2e_unknown_model_is_reported(fake_ollama):
    fake_ollama.fail["/api/chat"] = (404, "model 'nope' not found")
    result = run_script("--host", fake_ollama.host, "--model", "nope")
    assert result.returncode == 1
    assert result.stderr.strip() == "Error: model 'nope' not found"


@pytest.mark.integration
def test_e2e_unreachable_ollama_is_reported():
    result = run_script("--host", "http://127.0.0.1:9")
    assert result.returncode == 1
    assert result.stderr.startswith("Error: could not reach Ollama at http://127.0.0.1:9")


@pytest.mark.integration
def test_e2e_help_lists_options():
    result = run_script("--help")
    assert result.returncode == 0
    for option in ("--pdf", "--question", "--model", "--embedding-model", "--host", "--trace"):
        assert option in result.stdout


# --- against the real Ollama ------------------------------------------------

def _ollama_has(*models):
    try:
        names = {m["model"].split(":")[0] for m in ollama.Client(timeout=2).list()["models"]}
    except Exception:
        return False
    return all(m in names for m in models)


@pytest.fixture(scope="module")
def live_run():
    """One real run with --trace on BOI.pdf (a few minutes); shared by the live tests."""
    return run_script("--trace")


requires_models = pytest.mark.skipif(not _ollama_has("llama3.2", "nomic-embed-text"),
                                     reason="Ollama with llama3.2 and nomic-embed-text is not available")


@pytest.mark.live
@requires_models
def test_live_e2e_answers_how_to_report_boi(live_run):
    assert live_run.returncode == 0, live_run.stderr
    answer = live_run.stdout.split("unique chunks sent to the LLM as context")[-1].lower()
    assert "online" in answer or "e-filing" in answer or "boiefiling" in answer, answer


@pytest.mark.live
@requires_models
def test_live_e2e_is_quiet(live_run):
    assert live_run.stderr == ""


@pytest.mark.live
@requires_models
def test_live_e2e_sends_only_real_questions_to_chroma(live_run):
    queries = re.findall(r"CHROMA QUERY: (.*)", live_run.stdout)
    generated = queries[:-1]  # the last search is the original question
    assert queries[-1] == "how to report BOI?"
    assert 1 <= len(generated) <= 5, queries
    assert not any(q.lower().startswith(("here are", "these")) for q in generated), queries


@pytest.mark.live
@requires_models
def test_live_e2e_keeps_the_boi_meaning(live_run):
    queries = " ".join(re.findall(r"CHROMA QUERY: (.*)", live_run.stdout)).lower()
    assert "bilateral" not in queries
    assert "beneficial ownership" in queries
