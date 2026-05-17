"""
AI Research Assistant RAG

A starter Retrieval-Augmented Generation project using LangChain,
FAISS vector search, and OpenAI-powered answer generation.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


DATA_FILE = Path("data/sample_document.txt")


def load_documents():
    loader = TextLoader(str(DATA_FILE), encoding="utf-8")
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(vector_store, query):
    results = vector_store.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results])


def generate_answer(query, context):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = f"""
You are a research assistant. Answer the question using only the context below.

Question:
{query}

Context:
{context}

Answer:
"""

    response = model.invoke(prompt)
    return response.content


def save_output(query, answer):
    output_path = Path("outputs/research_answer.md")
    output_path.parent.mkdir(exist_ok=True)

    output_path.write_text(
        f"# Research Answer\n\n## Question\n{query}\n\n## Answer\n{answer}\n",
        encoding="utf-8"
    )


def main():
    query = "What is Retrieval-Augmented Generation?"

    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)

    context = retrieve_context(vector_store, query)
    answer = generate_answer(query, context)

    print("Question:")
    print(query)
    print("\nAnswer:")
    print(answer)

    save_output(query, answer)


if __name__ == "__main__":
    main()