import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.document_loader import (
    process_all_pdfs
)

from src.chunking import (
    split_documents
)

from src.embeddings import (
    EmbeddingManager
)

from src.vector_store import (
    VectorStore
)

from src.retriever import (
    RAGRetriever
)

from src.rag_pipeline import (
    rag_simple
)


def main():

    # ==========================================
    # 1. Load PDFs
    # ==========================================

    documents = process_all_pdfs(
        "./data/pdfs"
    )

    if not documents:

        print(
            "No PDF files found."
        )

        return

    # ==========================================
    # 2. Split documents
    # ==========================================

    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    # ==========================================
    # 3. Create embedding manager
    # ==========================================

    embedding_manager = (
        EmbeddingManager(
            "all-MiniLM-L6-v2"
        )
    )

    # ==========================================
    # 4. Extract text
    # ==========================================

    texts = [
        doc.page_content
        for doc in chunks
    ]

    # ==========================================
    # 5. Generate embeddings
    # ==========================================

    embeddings = (
        embedding_manager
        .generate_embeddings(texts)
    )

    # ==========================================
    # 6. Create vector store
    # ==========================================

    vector_store = VectorStore(
        collection_name="pdf_documents",
        persist_directory="./vector_store"
    )

    # ==========================================
    # 7. Store chunks and embeddings
    # ==========================================

    vector_store.add_documents(
        chunks,
        embeddings
    )

    # ==========================================
    # 8. Create retriever
    # ==========================================

    rag_retriever = RAGRetriever(
        vector_store,
        embedding_manager
    )

    # ==========================================
    # 9. Load API key
    # ==========================================

    load_dotenv()

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:

        raise ValueError(
            "GROQ_API_KEY not found in .env"
        )

    # ==========================================
    # 10. Create LLM
    # ==========================================

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-it",
        temperature=0.1,
        max_tokens=1024
    )

    # ==========================================
    # 11. Ask question
    # ==========================================

    question = (
        "What is the attention mechanism?"
    )

    answer = rag_simple(
        question,
        rag_retriever,
        llm,
        top_k=3
    )

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(answer)


if __name__ == "__main__":
    main()