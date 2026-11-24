import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_answer(question, document_text, use_general_knowledge=False):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to your .env file."
        )

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 90000},
    )

    if use_general_knowledge:
        instruction = (
            "You are an AI document assistant.\n"
            "First, answer using the document text if it contains relevant information.\n"
            "If the document does not contain enough information, you may use general knowledge.\n"
            "Clearly state when an answer comes from general knowledge rather than the document."
        )
    else:
        instruction = (
            "You are an AI document assistant.\n"
            "Answer the user's question using only the document text below.\n"
            "If the answer is not in the document, say that clearly."
        )

    prompt = (
        f"{instruction}\n\n"
        "Document text:\n"
        f"{document_text}\n\n"
        "Question:\n"
        f"{question}"
    )

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    return interaction.output_text