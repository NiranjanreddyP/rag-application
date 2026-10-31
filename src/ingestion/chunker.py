
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")
    return chunks
