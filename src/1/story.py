"""Stream generated text from a local Ollama server."""

import argparse
import json
import sys

import requests
from requests import RequestException

DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"
DEFAULT_PROMPT = "tell me a short story and make it funny."
DEFAULT_TIMEOUT = 300  # seconds to wait for the server between chunks


class OllamaError(Exception):
    """The Ollama request failed: unreachable server, HTTP error or bad reply."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def build_payload(model, prompt, stream=True):
    if not model or not model.strip():
        raise ValueError("model must not be empty")
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    return {"model": model, "prompt": prompt, "stream": stream}


def parse_chunk(line):
    """Return the text in one NDJSON line of an Ollama reply ("" for blank lines)."""
    if line is None:
        return ""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    if not line.strip():
        return ""
    try:
        result = json.loads(line)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Invalid JSON from Ollama: {line!r}") from exc
    if "error" in result:
        raise OllamaError(result["error"])
    return result.get("response", "")


def stream_generate(url, model, prompt, session=None, timeout=DEFAULT_TIMEOUT):
    """Yield the generated text piece by piece as Ollama streams it."""
    session = session or requests
    payload = build_payload(model, prompt)
    try:
        response = session.post(url, json=payload, stream=True, timeout=timeout)
    except RequestException as exc:
        raise OllamaError(f"Could not reach Ollama at {url}: {exc}") from exc

    if response.status_code != 200:
        raise OllamaError(f"{response.status_code} {response.text}", response.status_code)

    for line in response.iter_lines():
        text = parse_chunk(line)
        if text:
            yield text


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Generate text with a local Ollama model.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"generate endpoint (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt to send")
    return parser.parse_args(argv)


def main(argv=None, out=None, err=None):
    out = out or sys.stdout
    err = err or sys.stderr
    args = parse_args(argv)

    try:
        chunks = stream_generate(args.url, args.model, args.prompt, session=requests)
        # Fetch the first chunk before printing anything, so a failed request
        # shows only the error, not a dangling "Generated Text:".
        first = next(chunks, "")
        print("Generated Text:", end=" ", file=out, flush=True)
        print(first, end="", file=out, flush=True)
        for text in chunks:
            print(text, end="", file=out, flush=True)
        print(file=out)
    except (OllamaError, ValueError) as exc:
        print(f"Error: {exc}", file=err)
        return 1
    return 0
