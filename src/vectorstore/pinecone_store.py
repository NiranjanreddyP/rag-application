
import os
import hashlib

from pinecone import Pinecone, EmbedQuery


NAMESPACE = "medical-rag"


def get_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.index(os.getenv("PINECONE_INDEX_NAME"))


def store_chunks(chunks):
    index = get_index()
    documents = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "document")
        page = chunk.metadata.get("page", 0)
        content = chunk.page_content

        # Stable ID to avoid duplicate records on re-ingestion
        unique_text = f"{source}|{page}|{content}"
        record_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        documents.append({
            "_id": record_id,
            "chunk_text": content,
            "title": f"{os.path.basename(source)} - Page {page}"
        })

    batch_size = 96

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]

        index.documents.upsert(
            namespace=NAMESPACE,
            documents=batch
        )

        print(f"Uploaded batch {(start // batch_size) + 1}")

    print(f"Total chunks uploaded: {len(documents)}")


def _get_field(match, field):
    # Support both dictionary and SDK object response formats
    if isinstance(match, dict):
        fields = match.get("fields") or {}
        return match.get(field, fields.get(field, ""))

    value = getattr(match, field, None)

    if value is None:
        fields = getattr(match, "fields", None)
        if fields is not None:
            value = fields.get(field, "")

    return value if value is not None else ""


def retrieve_chunks(question, top_k=3):
    index = get_index()

    results = index.documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[
            EmbedQuery(
                field="chunk_text",
                query=question
            )
        ],
        include_fields=["chunk_text", "title"]
    )

    retrieved_chunks = []

    for match in results.matches:
        retrieved_chunks.append({
            "content": _get_field(match, "chunk_text"),
            "title": _get_field(match, "title"),
            "score": getattr(match, "score", None)
        })

    return retrieved_chunks
