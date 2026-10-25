"""Unit tests: no network, the Ollama client is replaced by a fake."""

import io

import ollama
import pytest

from ollama_basics import (
    DEFAULT_BASE_MODEL,
    DEFAULT_HOST,
    DEFAULT_MODEL_NAME,
    DEFAULT_QUESTION,
    KNOWITALL_SYSTEM,
    KNOWITALL_TEMPERATURE,
    ask,
    ask_stream,
    create_model,
    generate,
    list_model_names,
    main,
    temporary_model,
    user_message,
)


class FakeClient:
    """Duck-typed stand-in for ollama.Client that records every call."""

    def __init__(self, models=("llama3.2:latest",), answer="Because of salt.",
                 stream_chunks=("Salt ", "from ", "rocks."), error=None, fail_on=None):
        self.models = list(models)
        self.answer = answer
        self.stream_chunks = list(stream_chunks)
        self.error = error
        self.fail_on = fail_on
        self.calls = []

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error and (self.fail_on is None or self.fail_on == name):
            raise self.error

    def list(self):
        self._record("list")
        return {"models": [{"model": m} for m in self.models]}

    def chat(self, model, messages, stream=False):
        self._record("chat", model=model, messages=messages, stream=stream)
        if stream:
            return iter([{"message": {"role": "assistant", "content": c}} for c in self.stream_chunks])
        return {"message": {"role": "assistant", "content": self.answer}}

    def generate(self, model, prompt):
        self._record("generate", model=model, prompt=prompt)
        return {"response": self.answer}

    def create(self, model, from_, system, parameters):
        self._record("create", model=model, from_=from_, system=system, parameters=parameters)
        self.models.append(f"{model}:latest")
        return {"status": "success"}

    def delete(self, model):
        self._record("delete", model=model)
        self.models.remove(f"{model}:latest")
        return {"status": "success"}

    def names(self):
        return [name for name, _ in self.calls]


# --- defaults -------------------------------------------------------------

def test_defaults_match_original_script():
    assert DEFAULT_HOST == "http://localhost:11434"
    assert DEFAULT_BASE_MODEL == "llama3.2"
    assert DEFAULT_MODEL_NAME == "knowitall"
    assert DEFAULT_QUESTION == "why is the ocean so salty?"
    assert KNOWITALL_SYSTEM == ("You are very smart assistant who knows everything about oceans. "
                                "You are very succinct and informative.")
    assert KNOWITALL_TEMPERATURE == 0.1


# --- user_message ---------------------------------------------------------

def test_user_message_builds_chat_message():
    assert user_message("hi") == {"role": "user", "content": "hi"}


@pytest.mark.parametrize("text", ["", "   ", None])
def test_user_message_rejects_blank_text(text):
    with pytest.raises(ValueError):
        user_message(text)


# --- list_model_names -----------------------------------------------------

def test_list_model_names_returns_names():
    client = FakeClient(models=["llama3.2:latest", "llava:7b"])
    assert list_model_names(client) == ["llama3.2:latest", "llava:7b"]


def test_list_model_names_empty():
    assert list_model_names(FakeClient(models=[])) == []


# --- ask / ask_stream -----------------------------------------------------

def test_ask_sends_one_user_message_and_returns_answer():
    client = FakeClient(answer="Rayleigh scattering.")
    assert ask(client, "llama3.2", "why is the sky blue?") == "Rayleigh scattering."
    assert client.calls == [("chat", {"model": "llama3.2",
                                      "messages": [{"role": "user", "content": "why is the sky blue?"}],
                                      "stream": False})]


def test_ask_stream_yields_chunks_in_order():
    client = FakeClient(stream_chunks=["Salt ", "", "from ", "rocks."])
    assert list(ask_stream(client, "llama3.2", "why salty?")) == ["Salt ", "from ", "rocks."]
    assert client.calls[0][1]["stream"] is True


def test_ask_rejects_blank_question_without_calling_ollama():
    client = FakeClient()
    with pytest.raises(ValueError):
        ask(client, "llama3.2", " ")
    assert client.calls == []


# --- generate -------------------------------------------------------------

def test_generate_returns_response_text():
    client = FakeClient(answer="Blue light scatters more.")
    assert generate(client, "llama3.2", "why is the sky blue?") == "Blue light scatters more."
    assert client.calls == [("generate", {"model": "llama3.2", "prompt": "why is the sky blue?"})]


# --- create_model / temporary_model ---------------------------------------

def test_create_model_uses_current_create_api():
    client = FakeClient()
    create_model(client, "knowitall", "llama3.2", "Be brief.", 0.1)
    assert client.calls == [("create", {"model": "knowitall", "from_": "llama3.2",
                                        "system": "Be brief.", "parameters": {"temperature": 0.1}})]


def test_temporary_model_creates_then_deletes():
    client = FakeClient()
    with temporary_model(client, "knowitall", "llama3.2", "Be brief.", 0.1) as name:
        assert name == "knowitall"
        assert "knowitall:latest" in client.models
    assert client.names() == ["create", "delete"]
    assert "knowitall:latest" not in client.models


def test_temporary_model_deletes_even_when_body_fails():
    client = FakeClient()
    with pytest.raises(RuntimeError):
        with temporary_model(client, "knowitall", "llama3.2", "Be brief.", 0.1):
            raise RuntimeError("boom")
    assert client.names() == ["create", "delete"]


def test_temporary_model_does_not_delete_if_create_failed():
    client = FakeClient(error=ollama.ResponseError("base model not found", 404), fail_on="create")
    with pytest.raises(ollama.ResponseError):
        with temporary_model(client, "knowitall", "nope", "Be brief.", 0.1):
            pass
    assert client.names() == ["create"]


# --- main (CLI) -----------------------------------------------------------

def test_main_runs_knowitall_flow_and_prints_answer():
    client = FakeClient(answer="Rivers carry dissolved minerals.")
    out, err = io.StringIO(), io.StringIO()

    assert main([], client=client, out=out, err=err) == 0
    assert out.getvalue() == "Rivers carry dissolved minerals.\n"
    assert err.getvalue() == ""
    assert client.names() == ["create", "generate", "delete"]
    create_args = client.calls[0][1]
    assert create_args == {"model": DEFAULT_MODEL_NAME, "from_": DEFAULT_BASE_MODEL,
                           "system": KNOWITALL_SYSTEM,
                           "parameters": {"temperature": KNOWITALL_TEMPERATURE}}
    assert client.calls[1][1] == {"model": DEFAULT_MODEL_NAME, "prompt": DEFAULT_QUESTION}


def test_main_accepts_options():
    client = FakeClient()
    main(["--base-model", "llama4", "--name", "oceanbot", "--question", "how deep is it?"],
         client=client, out=io.StringIO(), err=io.StringIO())
    assert client.calls[0][1]["from_"] == "llama4"
    assert client.calls[0][1]["model"] == "oceanbot"
    assert client.calls[1][1] == {"model": "oceanbot", "prompt": "how deep is it?"}


def test_main_reports_ollama_error_and_still_deletes_model():
    client = FakeClient(error=ollama.ResponseError("out of memory", 500), fail_on="generate")
    err = io.StringIO()

    assert main([], client=client, out=io.StringIO(), err=err) == 1
    assert err.getvalue() == "Error: out of memory\n"
    assert client.names() == ["create", "generate", "delete"]


def test_main_reports_connection_error():
    client = FakeClient(error=ConnectionError("Failed to connect to Ollama"), fail_on="create")
    err = io.StringIO()

    assert main([], client=client, out=io.StringIO(), err=err) == 1
    assert err.getvalue() == "Error: Failed to connect to Ollama\n"
