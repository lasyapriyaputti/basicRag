def rag_simple(
    query,
    retriever,
    llm,
    top_k=3
):
    """Retrieve context and generate an answer."""

    # Retrieve relevant chunks
    results = retriever.retrieve(
        query,
        top_k=top_k
    )

    # Extract chunk text
    context = (
        "\n\n".join(
            [
                doc["content"]
                for doc in results
            ]
        )
        if results
        else ""
    )

    if not context:

        return (
            "No relevant context found."
        )

    # Create prompt
    prompt = f"""
Use the following context to answer
the question concisely.

Context:
{context}

Question:
{query}

Answer:
"""

    # Send prompt to LLM
    response = llm.invoke(prompt)

    return response.content