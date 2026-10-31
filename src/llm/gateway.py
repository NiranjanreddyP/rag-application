import os
from groq import Groq


def generate_answer(question, context):

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
                You are a medical information assistant.
                Answer only using the provided context.
                If information is unavailable, say so.
                Do not invent medical facts.
                """
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {question}
                """
            }
        ],
        temperature=0.2,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content