
from src.vectorstore.pinecone_store import retrieve_chunks
from src.llm.gateway import generate_answer


def prepare_context(retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
            Context {i}:
            {chunk['content']}
            Source: {chunk['title']}
            """
        )

    return "\n".join(context_parts)


def rag_pipeline(question):

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(
        question,
        top_k=3
    )

    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information in the document.",
            "sources": []
        }

    # Step 2: Prepare context
    context = prepare_context(retrieved_chunks)

    # Step 3: Generate answer through LiteLLM
    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer,
        "sources": [
            {
                "title": chunk["title"],
                "score": chunk["score"]
            }
            for chunk in retrieved_chunks
        ]
    }
