"""
AI Research Assistant RAG

A starter research assistant that retrieves relevant text from a small
knowledge base and generates a simple source-aware response.
"""


DOCUMENTS = [
    {
        "title": "LangChain Overview",
        "content": "LangChain helps developers build applications using language models, prompts, chains, tools, and retrieval workflows."
    },
    {
        "title": "RAG Overview",
        "content": "Retrieval-Augmented Generation combines document retrieval with language generation to produce more grounded responses."
    },
    {
        "title": "LangGraph Overview",
        "content": "LangGraph is useful for building controlled stateful workflows, agentic systems, and multi-step automation."
    }
]


def retrieve_documents(query):
    query_words = set(query.lower().split())
    results = []

    for document in DOCUMENTS:
        content_words = set(document["content"].lower().split())
        score = len(query_words.intersection(content_words))

        if score > 0:
            results.append((score, document))

    results.sort(reverse=True, key=lambda item: item[0])
    return [document for score, document in results]


def generate_answer(query, retrieved_docs):
    if not retrieved_docs:
        return "No relevant context found."

    context = retrieved_docs[0]

    return (
        f"Question: {query}\n\n"
        f"Answer: Based on the retrieved document, {context['content']}\n\n"
        f"Source: {context['title']}"
    )


def main():
    query = "What is RAG in AI applications?"

    retrieved_docs = retrieve_documents(query)
    answer = generate_answer(query, retrieved_docs)

    print(answer)


if __name__ == "__main__":
    main()