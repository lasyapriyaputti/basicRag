
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def process_all_pdfs(pdf_directory):
    """Load all PDF files from a directory."""

    all_documents = []
    pdf_dir = Path(pdf_directory).resolve()

    if not pdf_dir.exists():
        print(f"PDF directory not found: {pdf_dir}")
        return all_documents

    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            source_path = pdf_file.relative_to(
                pdf_dir
            ).as_posix()

            for doc in documents:
                doc.metadata["source_file"] = source_path
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents
