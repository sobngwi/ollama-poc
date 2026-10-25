"""Unit tests: no network, the Ollama client is replaced by a fake."""

import io

import ollama
import pytest

from grocery import (
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    OTHER_HEADING,
    SCENARIO_DIR,
    add_missing_items,
    build_prompt,
    categorize,
    find_missing_items,
    main,
    read_items,
    save_text,
)

CATEGORIZED = """**Produce**
- Apples
- Carrots

**Dairy**
- Milk"""


class FakeClient:
    def __init__(self, response=CATEGORIZED, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, model, prompt, options=None):
        self.calls.append({"model": model, "prompt": prompt, "options": options})
        if self.error:
            raise self.error
        return {"response": self.response}


@pytest.fixture
def grocery_file(tmp_path):
    path = tmp_path / "grocery_list.txt"
    path.write_text("Apples\nMilk\nCarrots\n")
    return path


# --- defaults -------------------------------------------------------------

def test_defaults_point_to_data_folder_next_to_the_script():
    assert DEFAULT_MODEL == "llama3.2"
    assert DEFAULT_INPUT == SCENARIO_DIR / "data" / "grocery_list.txt"
    assert DEFAULT_OUTPUT == SCENARIO_DIR / "data" / "categorized_grocery_list.txt"


def test_shipped_grocery_list_is_present():
    items = read_items(DEFAULT_INPUT)
    assert len(items) == 49
    assert items[0] == "Apples" and items[-1] == "Toothpaste"


# --- read_items -----------------------------------------------------------

def test_read_items_returns_one_item_per_line(grocery_file):
    assert read_items(grocery_file) == ["Apples", "Milk", "Carrots"]


def test_read_items_strips_spaces_and_skips_blank_lines(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("\n  Apples  \n\n\tMilk\n   \n")
    assert read_items(path) == ["Apples", "Milk"]


def test_read_items_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_items(tmp_path / "nope.txt")


def test_read_items_empty_file_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n  \n")
    with pytest.raises(ValueError, match="no grocery items"):
        read_items(path)


# --- build_prompt ---------------------------------------------------------

def test_build_prompt_lists_every_item_on_its_own_line():
    prompt = build_prompt(["Apples", "Milk"])
    assert "Here is a list of grocery items:\n\nApples\nMilk\n" in prompt


def test_build_prompt_keeps_original_instructions():
    prompt = build_prompt(["Apples"])
    assert prompt.startswith("You are an assistant that categorizes and sorts grocery items.")
    assert "Produce, Dairy, Meat, Bakery, Beverages" in prompt
    assert "Sort the items alphabetically within each category." in prompt
    assert "bullet points or numbering" in prompt


def test_build_prompt_rejects_empty_list():
    with pytest.raises(ValueError):
        build_prompt([])


# --- find_missing_items / add_missing_items --------------------------------

def test_find_missing_items_none_missing():
    assert find_missing_items(["Apples", "Milk", "Carrots"], CATEGORIZED) == []


def test_find_missing_items_keeps_input_order():
    assert find_missing_items(["Toothpaste", "Apples", "Oatmeal"], CATEGORIZED) == ["Toothpaste", "Oatmeal"]


def test_find_missing_items_ignores_case():
    assert find_missing_items(["apples", "MILK"], CATEGORIZED) == []


def test_find_missing_items_matches_whole_words_only():
    # "Butter" inside "Peanut Butter" must not count as present.
    assert find_missing_items(["Butter"], "- Peanut Butter") == ["Butter"]
    assert find_missing_items(["Tea"], "- Steak") == ["Tea"]


def test_add_missing_items_appends_other_section():
    text = add_missing_items(CATEGORIZED, ["Toothpaste", "Oatmeal"])
    assert text == f"{CATEGORIZED}\n\n{OTHER_HEADING}\n- Oatmeal\n- Toothpaste"


def test_add_missing_items_leaves_text_alone_when_nothing_missing():
    assert add_missing_items(CATEGORIZED, []) == CATEGORIZED


# --- categorize -----------------------------------------------------------

def test_categorize_sends_prompt_to_model_and_returns_stripped_text():
    client = FakeClient(response="\n\n" + CATEGORIZED + "\n\n")
    assert categorize(client, "llama3.2", ["Apples", "Milk"]) == CATEGORIZED
    assert client.calls[0]["model"] == "llama3.2"
    assert client.calls[0]["prompt"] == build_prompt(["Apples", "Milk"])


def test_categorize_uses_temperature_zero_so_items_are_not_dropped():
    client = FakeClient()
    categorize(client, "llama3.2", ["Apples"])
    assert client.calls[0]["options"] == {"temperature": 0}


def test_categorize_adds_items_the_model_dropped():
    client = FakeClient(response=CATEGORIZED)
    text = categorize(client, "llama3.2", ["Apples", "Milk", "Carrots", "Toothpaste"])
    assert text.endswith(f"{OTHER_HEADING}\n- Toothpaste")
    assert find_missing_items(["Apples", "Milk", "Carrots", "Toothpaste"], text) == []


def test_categorize_empty_answer_raises():
    with pytest.raises(ValueError, match="empty answer"):
        categorize(FakeClient(response="   "), "llama3.2", ["Apples"])


# --- save_text ------------------------------------------------------------

def test_save_text_writes_file_and_creates_folders(tmp_path):
    path = tmp_path / "out" / "nested" / "result.txt"
    save_text(path, "hello")
    assert path.read_text() == "hello\n"


def test_save_text_overwrites_existing_file(tmp_path):
    path = tmp_path / "result.txt"
    path.write_text("old content")
    save_text(path, "new")
    assert path.read_text() == "new\n"


# --- main (CLI) -----------------------------------------------------------

def test_main_prints_list_saves_file_and_returns_zero(grocery_file, tmp_path):
    output = tmp_path / "categorized.txt"
    out, err = io.StringIO(), io.StringIO()

    code = main(["--input", str(grocery_file), "--output", str(output)],
                client=FakeClient(), out=out, err=err)

    assert code == 0
    assert out.getvalue() == (
        "==== Categorized List: =====\n\n"
        f"{CATEGORIZED}\n\n"
        f"Categorized grocery list has been saved to '{output}'.\n"
    )
    assert err.getvalue() == ""
    assert output.read_text() == CATEGORIZED + "\n"


def test_main_uses_model_option(grocery_file, tmp_path):
    client = FakeClient()
    main(["--input", str(grocery_file), "--output", str(tmp_path / "o.txt"), "--model", "llama4"],
         client=client, out=io.StringIO(), err=io.StringIO())
    assert client.calls[0]["model"] == "llama4"


def test_main_missing_input_file_returns_one_without_calling_ollama(tmp_path):
    client = FakeClient()
    err = io.StringIO()
    missing = tmp_path / "missing.txt"

    assert main(["--input", str(missing), "--output", str(tmp_path / "o.txt")],
                client=client, out=io.StringIO(), err=err) == 1
    assert err.getvalue() == f"Error: input file '{missing}' not found.\n"
    assert client.calls == []


def test_main_empty_input_file_returns_one(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    err = io.StringIO()
    assert main(["--input", str(empty), "--output", str(tmp_path / "o.txt")],
                client=FakeClient(), out=io.StringIO(), err=err) == 1
    assert "no grocery items" in err.getvalue()


def test_main_ollama_error_returns_one_and_writes_nothing(grocery_file, tmp_path):
    output = tmp_path / "o.txt"
    err = io.StringIO()
    client = FakeClient(error=ollama.ResponseError("model 'x' not found", 404))

    assert main(["--input", str(grocery_file), "--output", str(output)],
                client=client, out=io.StringIO(), err=err) == 1
    assert err.getvalue() == "Error: model 'x' not found\n"
    assert not output.exists()


def test_main_connection_error_returns_one(grocery_file, tmp_path):
    err = io.StringIO()
    client = FakeClient(error=ConnectionError("Failed to connect to Ollama"))
    assert main(["--input", str(grocery_file), "--output", str(tmp_path / "o.txt")],
                client=client, out=io.StringIO(), err=err) == 1
    assert err.getvalue() == "Error: Failed to connect to Ollama\n"


# --- Claude provider --------------------------------------------------------

from types import SimpleNamespace

import anthropic

from grocery import DEFAULT_MODELS, ClaudeClient


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic(): records beta.messages.create calls."""

    def __init__(self, text=CATEGORIZED, stop_reason="end_turn", error=None):
        self.kwargs = None
        self._text = text
        self._stop_reason = stop_reason
        self._error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if self._error:
            raise self._error
        content = [SimpleNamespace(type="thinking", thinking=""),
                   SimpleNamespace(type="text", text=self._text)]
        return SimpleNamespace(content=content, stop_reason=self._stop_reason, stop_details=None)


def test_default_models_per_provider():
    assert DEFAULT_MODELS == {"ollama": "llama3.2", "claude": "claude-opus-5-5"}


def test_claude_client_sends_prompt_and_returns_text_like_ollama():
    fake = FakeAnthropic()
    result = ClaudeClient(fake).generate(model="claude-opus-5-5", prompt="sort these",
                                         options={"temperature": 0})
    assert result == {"response": CATEGORIZED}
    assert fake.kwargs["model"] == "claude-opus-5-5"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "sort these"}]
    assert fake.kwargs["output_config"] == {"effort": "low"}
    assert fake.kwargs["fallbacks"] == "default"
    assert fake.kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    # Sampling options are Ollama-only; the Claude API rejects temperature.
    assert "temperature" not in fake.kwargs and "options" not in fake.kwargs


def test_claude_client_refusal_raises():
    fake = FakeAnthropic(stop_reason="refusal")
    with pytest.raises(ValueError, match="declined"):
        ClaudeClient(fake).generate(model="claude-opus-5-5", prompt="x")


def test_claude_client_works_with_categorize():
    text = categorize(ClaudeClient(FakeAnthropic()), "claude-opus-5-5", ["Apples", "Milk", "Toothpaste"])
    assert text.endswith(f"{OTHER_HEADING}\n- Toothpaste")


def test_main_claude_provider_uses_claude_default_model(grocery_file, tmp_path):
    fake = FakeAnthropic()
    code = main(["--provider", "claude", "--input", str(grocery_file), "--output", str(tmp_path / "o.txt")],
                client=ClaudeClient(fake), out=io.StringIO(), err=io.StringIO())
    assert code == 0
    assert fake.kwargs["model"] == "claude-opus-5-5"


def test_main_reports_anthropic_errors(grocery_file, tmp_path):
    fake = FakeAnthropic(error=anthropic.AnthropicError("invalid x-api-key"))
    err = io.StringIO()
    code = main(["--provider", "claude", "--input", str(grocery_file), "--output", str(tmp_path / "o.txt")],
                client=ClaudeClient(fake), out=io.StringIO(), err=err)
    assert code == 1
    assert err.getvalue() == "Error: invalid x-api-key\n"
