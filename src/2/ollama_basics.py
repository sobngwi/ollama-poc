"""Basic uses of the ollama Python library: list, chat, generate, create and delete models."""

import argparse
import sys
from contextlib import contextmanager

import ollama

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BASE_MODEL = "llama3.2"
DEFAULT_MODEL_NAME = "knowitall"
DEFAULT_QUESTION = "why is the ocean so salty?"
KNOWITALL_SYSTEM = ("You are very smart assistant who knows everything about oceans. "
                    "You are very succinct and informative.")
KNOWITALL_TEMPERATURE = 0.1


def user_message(text):
    if not text or not text.strip():
        raise ValueError("message text must not be empty")
    return {"role": "user", "content": text}


def list_model_names(client):
    return [m["model"] for m in client.list()["models"]]


def ask(client, model, question):
    """Ask one chat question and return the whole answer."""
    response = client.chat(model=model, messages=[user_message(question)], stream=False)
    return response["message"]["content"]


def ask_stream(client, model, question):
    """Ask one chat question and yield the answer piece by piece."""
    for chunk in client.chat(model=model, messages=[user_message(question)], stream=True):
        text = chunk["message"]["content"]
        if text:
            yield text


def generate(client, model, prompt):
    return client.generate(model=model, prompt=prompt)["response"]


def create_model(client, name, base_model, system, temperature):
    # ollama >= 0.4 replaced the `modelfile=` argument with these fields.
    return client.create(model=name, from_=base_model, system=system,
                         parameters={"temperature": temperature})


@contextmanager
def temporary_model(client, name, base_model, system, temperature):
    """Create a custom model for the duration of a `with` block, then delete it."""
    create_model(client, name, base_model, system, temperature)
    try:
        yield name
    finally:
        client.delete(name)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create a temporary ocean-expert model, ask it a question, then delete it.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server (default: {DEFAULT_HOST})")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help=f"model to build on (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME,
                        help=f"name of the temporary model (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="question to ask")
    return parser.parse_args(argv)


def main(argv=None, client=None, out=None, err=None):
    out = out or sys.stdout
    err = err or sys.stderr
    args = parse_args(argv)
    client = client or ollama.Client(host=args.host)

    try:
        with temporary_model(client, args.name, args.base_model,
                             KNOWITALL_SYSTEM, KNOWITALL_TEMPERATURE) as name:
            print(generate(client, name, args.question), file=out)
    except ollama.ResponseError as exc:
        print(f"Error: {exc.error}", file=err)
        return 1
    except (ConnectionError, ValueError) as exc:
        print(f"Error: {exc}", file=err)
        return 1
    return 0
