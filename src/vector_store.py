
import os
import hashlib
import numpy as np
import chromadb

from collections import defaultdict
from typing import List, Any


class VectorStore:

    def __init__(
        self,
        collection_name="pdf_documents",
        persist_directory="./vector_store"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF embeddings for RAG"
                }
            )
        )

        print("\nVector store initialized.")
        print("Existing records:", self.collection.count())

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray
    ):
        if len(documents) != len(embeddings):
            raise ValueError(
                "Documents and embeddings count must be the same"
            )

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        page_chunk_counts = defaultdict(int)

        for doc, embedding in zip(documents, embeddings):
            source = doc.metadata.get("source_file")

            if not source:
                raise ValueError(
                    "Every document chunk must have source_file metadata"
                )

            page = doc.metadata.get("page", 0)
            page_key = (source, page)

            chunk_index = page_chunk_counts[page_key]
            page_chunk_counts[page_key] += 1

            content = doc.page_content

            # Stable ID based on source, page, position and content.
            id_input = (
                f"{source}|{page}|{chunk_index}|{content}"
            )

            doc_id = hashlib.sha256(
                id_input.encode("utf-8")
            ).hexdigest()

            metadata = dict(doc.metadata)
            metadata["doc_index"] = chunk_index
            metadata["content_length"] = len(content)

            ids.append(doc_id)
            metadatas.append(metadata)
            documents_text.append(content)
            embeddings_list.append(embedding.tolist())

        # Insert new IDs or update existing IDs.
        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

        # Remove old IDs that are no longer part of the
        # current complete PDF indexing run.
        current_ids = set(ids)
        existing_ids = self.collection.get(
            include=["metadatas"]
        )

        stale_ids = [
            doc_id
            for doc_id in existing_ids["ids"]
            if doc_id not in current_ids
        ]

        if stale_ids:
            self.collection.delete(ids=stale_ids)

        print(f"Indexed {len(ids)} current chunks.")
        print(f"Removed {len(stale_ids)} stale records.")
        print("Total records:", self.collection.count())
