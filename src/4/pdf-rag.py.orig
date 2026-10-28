## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements-rag.txt to install the required packages

import logging
import os
import sys

# Quiet run: no Unstructured advertisement, no Chroma telemetry, and no pdfminer
# "Cannot set non-stroke color" warnings (harmless PDF color settings it can't parse).
os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# Configure logging first: unstructured_client later calls basicConfig(level=INFO),
# which would print every HTTP request; basicConfig is a no-op once logging is set up.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

doc_path = "./data/BOI.pdf"
model = "llama3.2"

# Local PDF file uploads (parsed on this machine, not sent to Unstructured's API)
if doc_path:
    loader = UnstructuredLoader(file_path=doc_path, languages=["eng"])
    # The loader returns one Document per paragraph/title; join them into one
    # document, as the old UnstructuredPDFLoader did, so chunking works the same.
    elements = loader.load()
    data = [Document(page_content="\n\n".join(e.page_content for e in elements),
                     metadata={"source": doc_path})]
    print("done loading....")
else:
    print("Upload a PDF file")

    # Preview first page
content = data[0].page_content
# print(content[:100])


# ==== End of PDF Ingestion ====


# ==== Extract Text from PDF Files and Split into Small Chunks ====

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("done splitting....")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

# ===== Add to vector database ===
import ollama

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database....")


## === Retrieval ===
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field

from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# set up our model to use
llm = ChatOllama(model=model)

# a simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
# The start of the document is included so the model reads acronyms (e.g. BOI) the way
# the document uses them instead of guessing another meaning.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    partial_variables={"document_start": content[:300]},
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search.

    The questions are about this document, which begins:
    \"\"\"{document_start}\"\"\"
    Keep every acronym and term with the meaning it has in this document.

    Return exactly five questions.
    Original question: {question}""",
)


class AlternativeQuestions(BaseModel):
    questions: list[str] = Field(description="Five rephrasings of the original question",
                                 min_length=1, max_length=5)


# Structured output: Ollama constrains the model to JSON matching AlternativeQuestions,
# so no intro/outro sentence ("Here are five alternative versions...") can reach the
# vector database. MultiQueryRetriever's default parser sends every line of free text.
generate_questions = (
    QUERY_PROMPT
    | llm.with_structured_output(AlternativeQuestions)
    | (lambda result: list(dict.fromkeys(q.strip() for q in result.questions if q.strip())))
)

retriever = MultiQueryRetriever(
    retriever=vector_db.as_retriever(),
    llm_chain=generate_questions,
    include_original=True,  # always search the user's own question too
)


# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ==== Tracing (python pdf-rag.py --trace) ====
from langchain_core.callbacks import BaseCallbackHandler


class VectorDBTracer(BaseCallbackHandler):
    """Print each query sent to the vector database and the chunks it returns.

    The MultiQueryRetriever is itself a retriever: it receives the original question,
    asks the LLM for rephrasings, sends each one to Chroma, then merges the results.
    Chroma searches are the retriever runs nested inside it.
    """

    def __init__(self, preview_chars=150):
        self.preview_chars = preview_chars
        self.chroma_runs = set()
        self.multi_query_runs = set()

    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id in self.multi_query_runs:
            self.chroma_runs.add(run_id)
            print(f"\n  >>> CHROMA QUERY: {query}")
        else:
            self.multi_query_runs.add(run_id)
            print(f"\n>>> ORIGINAL QUESTION: {query}")

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        if run_id in self.chroma_runs:
            print(f"  <<< {len(documents)} chunks returned:")
            for i, doc in enumerate(documents, 1):
                text = " ".join(doc.page_content.split())[: self.preview_chars]
                print(f"      [{i}] {text}...")
        else:
            print(f"\n<<< {len(documents)} unique chunks sent to the LLM as context\n")


callbacks = [VectorDBTracer()] if "--trace" in sys.argv[1:] else []


# Pass the question as a plain string (a one-item tuple only worked by accident).
# res = chain.invoke(input="what is the document about?")
"""res = chain.invoke(
    input="what are the main points as a business owner I should be aware of?"
)
"""
res = chain.invoke(input="how to report BOI?", config={"callbacks": callbacks})

print(res)
