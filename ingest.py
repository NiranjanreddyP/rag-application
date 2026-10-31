
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import create_chunks
from src.vectorstore.pinecone_store import store_chunks


def main():

    pdf_path = "./docs/rag1.pdf"

    # Step 1: Load PDF
    documents = load_pdf(pdf_path)

    # Step 2: Split into chunks
    chunks = create_chunks(documents)

    # Step 3: Store in Pinecone
    store_chunks(chunks)

    print("PDF ingestion completed successfully.")


if __name__ == "__main__":
    main()
