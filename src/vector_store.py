import os
import uuid
import numpy as np
import chromadb

from typing import List, Any


class VectorStore:

    def __init__(
        self,
        collection_name="pdf_documents",
        persist_directory="./vector_store"
    ):

        self.collection_name = (
            collection_name
        )

        self.persist_directory = (
            persist_directory
        )

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):

        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )

        self.client = (
            chromadb.PersistentClient(
                path=self.persist_directory
            )
        )

        self.collection = (
            self.client
            .get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description":
                    "PDF embeddings for RAG"
                }
            )
        )

        print(
            "\nVector store initialized."
        )

        print(
            "Existing documents:",
            self.collection.count()
        )

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray
    ):

        if len(documents) != len(embeddings):

            raise ValueError(
                "Documents and embeddings "
                "count must be the same"
            )

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            doc_id = (
                f"doc_{uuid.uuid4().hex[:8]}_{i}"
            )

            ids.append(doc_id)

            metadata = dict(
                doc.metadata
            )

            metadata["doc_index"] = i

            metadata["content_length"] = (
                len(doc.page_content)
            )

            metadatas.append(metadata)

            documents_text.append(
                doc.page_content
            )

            embeddings_list.append(
                embedding.tolist()
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )

        print(
            f"Added {len(documents)} chunks "
            f"to ChromaDB"
        )

        print(
            "Total documents:",
            self.collection.count()
        )