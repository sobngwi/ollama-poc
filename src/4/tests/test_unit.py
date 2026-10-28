"""Unit tests: no network, no model. Fake embeddings and fake LLMs stand in for Ollama."""

import io
import logging
import math
import os
import re

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from pdf_rag import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PDF,
    DEFAULT_QUESTION,
    NUM_QUESTIONS,
    SCENARIO_DIR,
    AlternativeQuestions,
    VectorDBTracer,
    build_chain,
    build_query_prompt,
    build_retriever,
    build_vector_db,
    clean_questions,
    configure_logging,
    load_pdf,
    main,
    make_question_generator,
    parse_args,
    split_document,
)


class KeywordEmbeddings(Embeddings):
    """Bag-of-words embeddings: texts sharing words are close, so retrieval is predictable."""

    DIM = 256

    def _embed(self, text):
        vector = [0.0] * self.DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[sum(map(ord, word)) % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeLoader:
    """Stands in for langchain_unstructured.UnstructuredLoader."""

    calls = []

    def __init__(self, file_path, **kwargs):
        FakeLoader.calls.append({"file_path": file_path, **kwargs})
        self.elements = FakeLoader.elements

    def load(self):
        return self.elements


CHUNKS = [
    Document(page_content="Reporting companies file the BOIR online at the BOI E-Filing portal."),
    Document(page_content="A passport or a driver's license can be used as an identifying document."),
    Document(page_content="Updated reports are due within 30 days of a change."),
]


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 placeholder")
    return path


@pytest.fixture
def fake_elements():
    FakeLoader.calls = []
    FakeLoader.elements = [Document(page_content="Title"), Document(page_content="Body text.")]
    return FakeLoader


# --- defaults -------------------------------------------------------------

def test_defaults():
    assert DEFAULT_PDF == SCENARIO_DIR / "data" / "BOI.pdf"
    assert DEFAULT_PDF.exists()
    assert DEFAULT_HOST == "http://localhost:11434"
    assert DEFAULT_MODEL == "llama3.2"
    assert DEFAULT_EMBEDDING_MODEL == "nomic-embed-text"
    assert DEFAULT_QUESTION == "how to report BOI?"
    assert (CHUNK_SIZE, CHUNK_OVERLAP, NUM_QUESTIONS) == (1200, 300, 5)


# --- load_pdf -------------------------------------------------------------

def test_load_pdf_joins_elements_into_one_document(pdf_file, fake_elements):
    document = load_pdf(pdf_file, loader_cls=fake_elements)
    assert document.page_content == "Title\n\nBody text."
    assert document.metadata == {"source": str(pdf_file)}


def test_load_pdf_reads_locally_in_english(pdf_file, fake_elements):
    load_pdf(pdf_file, loader_cls=fake_elements)
    assert fake_elements.calls == [{"file_path": str(pdf_file), "languages": ["eng"]}]


def test_load_pdf_missing_file_raises_before_loading(tmp_path, fake_elements):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf", loader_cls=fake_elements)
    assert fake_elements.calls == []


def test_load_pdf_without_text_raises(pdf_file, fake_elements):
    fake_elements.elements = [Document(page_content="   ")]
    with pytest.raises(ValueError, match="no text"):
        load_pdf(pdf_file, loader_cls=fake_elements)


# --- split_document -------------------------------------------------------

def test_split_document_respects_chunk_size_and_overlap():
    words = " ".join(f"word{i}" for i in range(400))
    chunks = split_document(Document(page_content=words, metadata={"source": "x"}),
                            chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata == {"source": "x"} for c in chunks)
    # consecutive chunks share text (the overlap)
    assert chunks[0].page_content.split()[-1] in chunks[1].page_content


def test_split_document_short_text_is_one_chunk():
    assert len(split_document(Document(page_content="short"))) == 1


# --- query prompt and questions --------------------------------------------

def test_query_prompt_includes_document_start_and_question():
    text = build_query_prompt("Beneficial Ownership Information (BOI)").format(question="how to report BOI?")
    assert "Beneficial Ownership Information (BOI)" in text
    assert "Original question: how to report BOI?" in text
    assert "meaning it has in this document" in text
    assert f"exactly {NUM_QUESTIONS} questions" in text


def test_alternative_questions_limits():
    assert AlternativeQuestions(questions=["a?"]).questions == ["a?"]
    with pytest.raises(ValidationError):
        AlternativeQuestions(questions=[])
    with pytest.raises(ValidationError):
        AlternativeQuestions(questions=[f"q{i}?" for i in range(NUM_QUESTIONS + 1)])


def test_clean_questions_strips_drops_blanks_and_duplicates_in_order():
    result = AlternativeQuestions(questions=[" How to file? ", "", "Where?", "How to file?", "  "])
    assert clean_questions(result) == ["How to file?", "Where?"]


def test_question_generator_sends_prompt_and_returns_clean_list():
    seen = []

    def fake_structured_llm(prompt_value):
        seen.append(prompt_value.to_string())
        return AlternativeQuestions(questions=["How to file BOI?", "How to file BOI?", "Where to file?"])

    generator = make_question_generator(RunnableLambda(fake_structured_llm), "Doc about BOI")
    assert generator.invoke({"question": "how to report BOI?"}) == ["How to file BOI?", "Where to file?"]
    assert "Doc about BOI" in seen[0] and "how to report BOI?" in seen[0]


# --- vector database and retriever ------------------------------------------

def test_vector_db_returns_the_matching_chunk():
    db = build_vector_db(CHUNKS, KeywordEmbeddings())
    best = db.similarity_search("which identifying document passport license", k=1)[0]
    assert "passport" in best.page_content


def test_each_vector_db_is_separate():
    # Chroma's in-memory store is shared by the whole process: reusing one collection
    # name would mix documents from different runs.
    first = build_vector_db(CHUNKS[:1], KeywordEmbeddings())
    second = build_vector_db(CHUNKS[1:], KeywordEmbeddings())
    assert len(first.get()["ids"]) == 1
    assert len(second.get()["ids"]) == 2


def test_retriever_searches_generated_questions_and_the_original():
    db = build_vector_db(CHUNKS, KeywordEmbeddings())
    generator = RunnableLambda(lambda _: ["which identifying document passport"])
    out = io.StringIO()

    docs = build_retriever(db, generator).invoke(
        "file BOIR online E-Filing", config={"callbacks": [VectorDBTracer(out=out)]})

    queries = re.findall(r"CHROMA QUERY: (.*)", out.getvalue())
    assert queries == ["which identifying document passport", "file BOIR online E-Filing"]
    contents = " ".join(d.page_content for d in docs)
    assert "passport" in contents and "E-Filing" in contents
    assert len(docs) == len({d.page_content for d in docs})  # unique union


# --- chain ----------------------------------------------------------------

def test_chain_answers_from_retrieved_context_only():
    db = build_vector_db(CHUNKS, KeywordEmbeddings())
    retriever = build_retriever(db, RunnableLambda(lambda _: []))
    prompts = []

    def fake_llm(prompt_value):
        prompts.append(prompt_value.to_string())
        return AIMessage(content="File it online.")

    answer = build_chain(retriever, RunnableLambda(fake_llm)).invoke("file BOIR online E-Filing")
    assert answer == "File it online."
    assert "Answer the question based ONLY on the following context" in prompts[0]
    assert "BOI E-Filing portal" in prompts[0]
    assert "Question: file BOIR online E-Filing" in prompts[0]


# --- tracer ---------------------------------------------------------------

def test_tracer_prints_original_question_queries_and_chunks():
    out = io.StringIO()
    tracer = VectorDBTracer(out=out, preview_chars=20)
    tracer.on_retriever_start({}, "how to report BOI?", run_id="multi")
    tracer.on_retriever_start({}, "how to file BOI?", run_id="chroma", parent_run_id="multi")
    tracer.on_retriever_end([Document(page_content="File   online\nat the portal today")], run_id="chroma")
    tracer.on_retriever_end(CHUNKS, run_id="multi")

    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert lines == [
        ">>> ORIGINAL QUESTION: how to report BOI?",
        "  >>> CHROMA QUERY: how to file BOI?",
        "  <<< 1 chunks returned:",
        "      [1] File online at the p...",  # whitespace collapsed, cut at 20 characters
        f"<<< {len(CHUNKS)} unique chunks sent to the LLM as context",
    ]


# --- logging and CLI --------------------------------------------------------

def test_noisy_libraries_are_silenced():
    configure_logging()
    assert os.environ["UNSTRUCTURED_DISABLE_NOTICE"] == "1"
    assert os.environ["ANONYMIZED_TELEMETRY"] == "False"
    assert logging.getLogger("pdfminer").level == logging.ERROR


def test_parse_args_defaults_and_options():
    args = parse_args([])
    assert (args.pdf, args.question, args.model, args.trace) == (DEFAULT_PDF, DEFAULT_QUESTION, DEFAULT_MODEL, False)
    args = parse_args(["--pdf", "x.pdf", "--question", "q?", "--model", "llama4",
                       "--embedding-model", "e", "--host", "http://h:1", "--trace"])
    assert (str(args.pdf), args.question, args.model, args.embedding_model, args.host, args.trace) == \
           ("x.pdf", "q?", "llama4", "e", "http://h:1", True)


def test_main_missing_pdf_reports_error_without_contacting_ollama(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    missing = tmp_path / "missing.pdf"
    # An unreachable host proves nothing is contacted: the error is about the file.
    code = main(["--pdf", str(missing), "--host", "http://127.0.0.1:9"], out=out, err=err)
    assert code == 1
    assert err.getvalue() == f"Error: PDF file '{missing}' not found.\n"
    assert out.getvalue() == ""


def test_main_pdf_without_text_reports_error(pdf_file, monkeypatch):
    import pdf_rag

    def empty_pdf(path):
        raise ValueError(f"no text found in PDF '{path}'")

    monkeypatch.setattr(pdf_rag, "load_pdf", empty_pdf)
    err = io.StringIO()
    assert main(["--pdf", str(pdf_file)], out=io.StringIO(), err=err) == 1
    assert err.getvalue() == f"Error: no text found in PDF '{pdf_file}'\n"
