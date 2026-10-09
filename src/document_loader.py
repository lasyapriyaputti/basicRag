from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def process_all_pdfs(pdf_directory):
    """Load all PDF files from a directory."""

    all_documents = []

    pdf_dir = Path(pdf_directory)

    pdf_files = list(
        pdf_dir.glob("**/*.pdf")
    )

    print(
        f"Found {len(pdf_files)} PDF files"
    )

    for pdf_file in pdf_files:

        print(
            f"\nProcessing: {pdf_file.name}"
        )

        try:

            loader = PyPDFLoader(
                str(pdf_file)
            )

            documents = loader.load()

            for doc in documents:

                doc.metadata["source_file"] = (
                    pdf_file.name
                )

                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)

            print(
                f"Loaded {len(documents)} pages"
            )

        except Exception as e:

            print(
                f"Error processing "
                f"{pdf_file.name}: {e}"
            )

    print(
        f"\nTotal pages loaded: "
        f"{len(all_documents)}"
    )

    return all_documents