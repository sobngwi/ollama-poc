"""Answer a question about a PDF with a local RAG pipeline (LangChain, Chroma, Ollama).

1. Load the PDF and split its text into overlapping chunks.
2. Embed the chunks with an Ollama embedding model into an in-memory Chroma store.
3. Ask the LLM for rephrasings of the question (structured output), search Chroma
   with each of them plus the original question, and merge the chunks found.
4. Answer the question from those chunks only.
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

# Quiet run: no Unstructured advertisement and no Chroma telemetry. Set before the
# libraries below are imported, since they read these at import time.
os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import ollama  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_classic.retrievers.multi_query import MultiQueryRetriever  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_unstructured import UnstructuredLoader  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

SCENARIO_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = SCENARIO_DIR / "data" / "BOI.pdf"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_QUESTION = "how to report BOI?"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
NUM_QUESTIONS = 5
DOCUMENT_START_CHARS = 300

# A simple technique to generate multiple questions from a single question and then
# retrieve documents based on those questions, getting the best of both worlds.
# The start of the document is included so the model reads acronyms (e.g. BOI) the way
# the document uses them instead of guessing another meaning.
QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate {num_questions}
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search.

The questions are about this document, which begins:
\"\"\"{document_start}\"\"\"
Keep every acronym and term with the meaning it has in this document.

Return exactly {num_questions} questions.
Original question: {question}"""

RAG_PROMPT = ChatPromptTemplate.from_template("""Answer the question based ONLY on the following context:
{context}
Question: {question}
""")


def configure_logging():
    """Silence the libraries' notices, telemetry and log noise."""
    os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    # Configure logging first: unstructured_client later calls basicConfig(level=INFO),
    # which would print every HTTP request; basicConfig is a no-op once logging is set up.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # pdfminer warns "Cannot set non-stroke color" for harmless PDF color settings.
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def load_pdf(path, loader_cls=UnstructuredLoader):
    """Read the PDF on this machine and return its text as one Document."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # strategy="fast" reads the PDF's text layer directly. With the default "auto",
    # Unstructured silently switches to OCR (which needs Tesseract) when a text check
    # fails, e.g. because its spaCy model is missing.
    # The loader returns one Document per paragraph/title; join them into one document
    # so the chunks are cut by size, not by the PDF's layout.
    elements = loader_cls(file_path=str(path), languages=["eng"], strategy="fast").load()
    text = "\n\n".join(e.page_content for e in elements if e.page_content.strip())
    if not text:
        raise ValueError(f"no text found in PDF '{path}'")
    return Document(page_content=text, metadata={"source": str(path)})


def split_document(document, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents([document])


def build_vector_db(chunks, embeddings, collection_name=None):
    # Chroma's in-memory store is shared by the whole process, so each database gets
    # its own collection; a fixed name would mix documents from earlier builds.
    collection_name = collection_name or f"simple-rag-{uuid.uuid4().hex[:8]}"
    return Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name=collection_name)


class AlternativeQuestions(BaseModel):
    questions: list[str] = Field(description=f"{NUM_QUESTIONS} rephrasings of the original question",
                                 min_length=1, max_length=NUM_QUESTIONS)


def build_query_prompt(document_start):
    return PromptTemplate(
        input_variables=["question"],
        partial_variables={"document_start": document_start, "num_questions": NUM_QUESTIONS},
        template=QUERY_TEMPLATE,
    )


def clean_questions(result):
    """Strip the questions and drop blanks and duplicates, keeping their order."""
    return list(dict.fromkeys(q.strip() for q in result.questions if q.strip()))


def make_question_generator(structured_llm, document_start):
    """Chain question -> list of rephrasings.

    `structured_llm` is `llm.with_structured_output(AlternativeQuestions)`: Ollama then
    constrains the model to that JSON, so no intro/outro sentence ("Here are five
    alternative versions...") can reach the vector database, as it does with
    MultiQueryRetriever's default line parser.
    """
    return build_query_prompt(document_start) | structured_llm | RunnableLambda(clean_questions)


def build_retriever(vector_db, question_generator):
    return MultiQueryRetriever(
        retriever=vector_db.as_retriever(),
        llm_chain=question_generator,
        include_original=True,  # always search the user's own question too
    )


def build_chain(retriever, llm):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


class VectorDBTracer(BaseCallbackHandler):
    """Print each query sent to the vector database and the chunks it returns.

    The MultiQueryRetriever is itself a retriever: it receives the original question,
    asks the LLM for rephrasings, sends each one to Chroma, then merges the results.
    Chroma searches are the retriever runs nested inside it.
    """

    def __init__(self, out=None, preview_chars=150):
        self.out = out or sys.stdout
        self.preview_chars = preview_chars
        self.chroma_runs = set()
        self.multi_query_runs = set()

    def _print(self, text):
        print(text, file=self.out)

    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id in self.multi_query_runs:
            self.chroma_runs.add(run_id)
            self._print(f"\n  >>> CHROMA QUERY: {query}")
        else:
            self.multi_query_runs.add(run_id)
            self._print(f"\n>>> ORIGINAL QUESTION: {query}")

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        if run_id in self.chroma_runs:
            self._print(f"  <<< {len(documents)} chunks returned:")
            for i, doc in enumerate(documents, 1):
                text = " ".join(doc.page_content.split())[: self.preview_chars]
                self._print(f"      [{i}] {text}...")
        else:
            self._print(f"\n<<< {len(documents)} unique chunks sent to the LLM as context\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Answer a question about a PDF with a local RAG pipeline.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF to read (default: data/BOI.pdf)")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help=f"question (default: {DEFAULT_QUESTION!r})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"chat model (default: {DEFAULT_MODEL})")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help=f"embedding model (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server (default: {DEFAULT_HOST})")
    parser.add_argument("--trace", action="store_true",
                        help="print each query sent to the vector database and the chunks it returns")
    return parser.parse_args(argv)


def main(argv=None, out=None, err=None):
    out = out or sys.stdout
    err = err or sys.stderr
    args = parse_args(argv)
    configure_logging()

    try:
        document = load_pdf(args.pdf)
        print("done loading....", file=out)
        chunks = split_document(document)
        print("done splitting....", file=out)

        ollama.Client(host=args.host).pull(args.embedding_model)
        embeddings = OllamaEmbeddings(model=args.embedding_model, base_url=args.host)
        vector_db = build_vector_db(chunks, embeddings)
        print("done adding to vector database....", file=out)

        llm = ChatOllama(model=args.model, base_url=args.host)
        question_generator = make_question_generator(
            llm.with_structured_output(AlternativeQuestions),
            document.page_content[:DOCUMENT_START_CHARS],
        )
        chain = build_chain(build_retriever(vector_db, question_generator), llm)
        callbacks = [VectorDBTracer(out=out)] if args.trace else []
        answer = chain.invoke(args.question, config={"callbacks": callbacks})
    except FileNotFoundError:
        print(f"Error: PDF file '{args.pdf}' not found.", file=err)
        return 1
    except ollama.ResponseError as exc:
        print(f"Error: {exc.error}", file=err)
        return 1
    except ConnectionError as exc:
        print(f"Error: could not reach Ollama at {args.host}: {exc}", file=err)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=err)
        return 1

    print(answer, file=out)
    return 0
