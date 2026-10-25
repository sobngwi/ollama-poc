"""Categorize and sort a grocery list with a local Ollama model."""

import argparse
import re
import sys
from pathlib import Path

import anthropic
import ollama

SCENARIO_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_MODELS = {"ollama": DEFAULT_MODEL, "claude": "claude-opus-5-5"}
DEFAULT_INPUT = SCENARIO_DIR / "data" / "grocery_list.txt"
DEFAULT_OUTPUT = SCENARIO_DIR / "data" / "categorized_grocery_list.txt"
OTHER_HEADING = "**Other**"

# Bullets, numbering and bold markers in front of an item: "- ", "• ", "3. ", "**".
LIST_MARKER = re.compile(r"^[\s\-*•·\d.)]*")

PROMPT_TEMPLATE = """You are an assistant that categorizes and sorts grocery items.

Here is a list of grocery items:

{items}

Please:

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.
"""


def read_items(path):
    """Return the non-blank lines of the file, stripped."""
    with open(path) as f:
        items = [line.strip() for line in f if line.strip()]
    if not items:
        raise ValueError(f"no grocery items in '{path}'")
    return items


def build_prompt(items):
    if not items:
        raise ValueError("no grocery items to categorize")
    return PROMPT_TEMPLATE.format(items="\n".join(items))


class ClaudeClient:
    """Gives the Claude API the same generate() shape as ollama.Client."""

    def __init__(self, client=None):
        self.client = client or anthropic.Anthropic()

    def generate(self, model, prompt, options=None):
        # `options` (temperature) is Ollama-only: the Claude API rejects sampling parameters.
        response = self.client.beta.messages.create(
            model=model,
            max_tokens=16000,
            output_config={"effort": "low"},  # sorting a list needs little reasoning
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if a safety classifier declines, retry on a fallback model
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise ValueError(f"model '{model}' declined to answer")
        text = "".join(block.text for block in response.content if block.type == "text")
        return {"response": text}


def make_client(provider, host):
    return ClaudeClient() if provider == "claude" else ollama.Client(host=host)


def find_missing_items(items, text):
    """Return the items that do not start any line of the model's list, in input order."""
    entries = [LIST_MARKER.sub("", line).lower() for line in text.splitlines()]
    return [item for item in items
            if not any(re.match(rf"{re.escape(item.lower())}\b", entry) for entry in entries)]


def add_missing_items(text, missing):
    if not missing:
        return text
    bullets = "\n".join(f"- {item}" for item in sorted(missing))
    return f"{text}\n\n{OTHER_HEADING}\n{bullets}"


def categorize(client, model, items):
    # Temperature 0 makes the answer repeatable, but llama3.2 still drops an
    # item or two from long lists, so anything it left out goes under "Other".
    response = client.generate(model=model, prompt=build_prompt(items), options={"temperature": 0})
    text = response["response"].strip()
    if not text:
        raise ValueError(f"model '{model}' returned an empty answer")
    return add_missing_items(text, find_missing_items(items, text))


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Categorize and sort a grocery list with Ollama or Claude.")
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="ollama",
                        help="where the model runs (default: ollama; claude needs ANTHROPIC_API_KEY)")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server (default: {DEFAULT_HOST})")
    parser.add_argument("--model", help=f"model to use (default: {DEFAULT_MODELS['ollama']} for ollama, "
                                        f"{DEFAULT_MODELS['claude']} for claude)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="grocery list, one item per line (default: data/grocery_list.txt)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="where to save the result (default: data/categorized_grocery_list.txt)")
    return parser.parse_args(argv)


def main(argv=None, client=None, out=None, err=None):
    out = out or sys.stdout
    err = err or sys.stderr
    args = parse_args(argv)
    model = args.model or DEFAULT_MODELS[args.provider]

    try:
        items = read_items(args.input)
        client = client or make_client(args.provider, args.host)
        text = categorize(client, model, items)
    except FileNotFoundError:
        print(f"Error: input file '{args.input}' not found.", file=err)
        return 1
    except ollama.ResponseError as exc:
        print(f"Error: {exc.error}", file=err)
        return 1
    except (ConnectionError, ValueError, anthropic.AnthropicError) as exc:
        print(f"Error: {exc}", file=err)
        return 1

    print("==== Categorized List: =====\n", file=out)
    print(text, file=out)
    print(file=out)
    save_text(args.output, text)
    print(f"Categorized grocery list has been saved to '{args.output}'.", file=out)
    return 0
