"""Unit tests: no network, the HTTP session is replaced by a fake."""

import io
import json

import pytest
import requests

import story
from story import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_URL,
    OllamaError,
    build_payload,
    main,
    parse_chunk,
    stream_generate,
)


class FakeResponse:
    def __init__(self, status_code=200, lines=(), text=""):
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text

    def iter_lines(self):
        return iter(self._lines)


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


def ndjson(*chunks, done=True):
    lines = [json.dumps({"response": c, "done": False}).encode() for c in chunks]
    if done:
        lines.append(json.dumps({"response": "", "done": True}).encode())
    return lines


# --- defaults -------------------------------------------------------------

def test_defaults_match_original_script():
    assert DEFAULT_URL == "http://localhost:11434/api/generate"
    assert DEFAULT_MODEL == "llama3.2"
    assert DEFAULT_PROMPT == "tell me a short story and make it funny."


# --- build_payload --------------------------------------------------------

def test_build_payload_contains_model_and_prompt():
    assert build_payload("m", "p") == {"model": "m", "prompt": "p", "stream": True}


def test_build_payload_can_disable_streaming():
    assert build_payload("m", "p", stream=False)["stream"] is False


@pytest.mark.parametrize("model, prompt", [("", "p"), ("m", ""), ("  ", "p"), ("m", "   ")])
def test_build_payload_rejects_blank_values(model, prompt):
    with pytest.raises(ValueError):
        build_payload(model, prompt)


# --- parse_chunk ----------------------------------------------------------

def test_parse_chunk_returns_response_text():
    assert parse_chunk(b'{"response": "Hello", "done": false}') == "Hello"


def test_parse_chunk_accepts_str():
    assert parse_chunk('{"response": "Hi"}') == "Hi"


def test_parse_chunk_decodes_utf8():
    assert parse_chunk('{"response": "café 🦙"}'.encode("utf-8")) == "café 🦙"


@pytest.mark.parametrize("line", [b"", b"   ", None])
def test_parse_chunk_ignores_blank_lines(line):
    assert parse_chunk(line) == ""


def test_parse_chunk_missing_response_key_gives_empty_string():
    assert parse_chunk(b'{"done": true}') == ""


def test_parse_chunk_reports_server_error_in_stream():
    with pytest.raises(OllamaError, match="model not found"):
        parse_chunk(b'{"error": "model not found"}')


def test_parse_chunk_invalid_json_raises_ollama_error():
    with pytest.raises(OllamaError, match="Invalid JSON"):
        parse_chunk(b"not json")


# --- stream_generate ------------------------------------------------------

def test_stream_generate_yields_text_chunks_in_order():
    session = FakeSession(FakeResponse(lines=ndjson("Once ", "upon ", "a time")))
    assert list(stream_generate("u", "m", "p", session=session)) == ["Once ", "upon ", "a time"]


def test_stream_generate_skips_empty_lines_and_empty_chunks():
    lines = [b"", *ndjson("A"), b""]
    session = FakeSession(FakeResponse(lines=lines))
    assert list(stream_generate("u", "m", "p", session=session)) == ["A"]


def test_stream_generate_posts_json_payload_with_streaming():
    session = FakeSession(FakeResponse(lines=ndjson("x")))
    list(stream_generate("http://h/api/generate", "m", "p", session=session, timeout=5))
    url, kwargs = session.calls[0]
    assert url == "http://h/api/generate"
    assert kwargs["json"] == {"model": "m", "prompt": "p", "stream": True}
    assert kwargs["stream"] is True
    assert kwargs["timeout"] == 5


def test_stream_generate_raises_on_http_error_with_status_and_body():
    session = FakeSession(FakeResponse(status_code=404, text='{"error":"model not found"}'))
    with pytest.raises(OllamaError) as excinfo:
        list(stream_generate("u", "m", "p", session=session))
    assert excinfo.value.status_code == 404
    assert "model not found" in str(excinfo.value)


def test_stream_generate_wraps_connection_errors():
    session = FakeSession(error=requests.ConnectionError("refused"))
    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        list(stream_generate("u", "m", "p", session=session))


# --- main (CLI) -----------------------------------------------------------

def test_main_prints_generated_text_and_returns_zero(monkeypatch):
    session = FakeSession(FakeResponse(lines=ndjson("Funny ", "story.")))
    monkeypatch.setattr(story, "requests", session)
    out, err = io.StringIO(), io.StringIO()

    assert main([], out=out, err=err) == 0
    assert out.getvalue() == "Generated Text: Funny story.\n"
    assert err.getvalue() == ""


def test_main_uses_defaults_when_no_arguments(monkeypatch):
    session = FakeSession(FakeResponse(lines=ndjson("x")))
    monkeypatch.setattr(story, "requests", session)
    main([], out=io.StringIO(), err=io.StringIO())
    url, kwargs = session.calls[0]
    assert url == DEFAULT_URL
    assert kwargs["json"]["model"] == DEFAULT_MODEL
    assert kwargs["json"]["prompt"] == DEFAULT_PROMPT


def test_main_accepts_url_model_and_prompt_options(monkeypatch):
    session = FakeSession(FakeResponse(lines=ndjson("x")))
    monkeypatch.setattr(story, "requests", session)
    main(["--url", "http://other/api/generate", "--model", "llama4", "--prompt", "hi"],
         out=io.StringIO(), err=io.StringIO())
    url, kwargs = session.calls[0]
    assert url == "http://other/api/generate"
    assert kwargs["json"]["model"] == "llama4"
    assert kwargs["json"]["prompt"] == "hi"


def test_main_reports_http_error_and_returns_one(monkeypatch):
    session = FakeSession(FakeResponse(status_code=500, text="boom"))
    monkeypatch.setattr(story, "requests", session)
    out, err = io.StringIO(), io.StringIO()

    assert main([], out=out, err=err) == 1
    assert err.getvalue() == "Error: 500 boom\n"


def test_main_reports_connection_error_and_returns_one(monkeypatch):
    session = FakeSession(error=requests.ConnectionError("refused"))
    monkeypatch.setattr(story, "requests", session)
    err = io.StringIO()

    assert main([], out=io.StringIO(), err=err) == 1
    assert err.getvalue().startswith("Error: Could not reach Ollama")
