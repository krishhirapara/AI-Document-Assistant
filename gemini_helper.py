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


def answer_from_context(question, retrieved_chunks):
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

    context_parts = []

    for chunk in retrieved_chunks:
        page_number = chunk["page_number"]
        text = chunk["text"]

        context_parts.append(
            f"Page {page_number}:\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""Answer the user's question using only the context below.

If the answer is not in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    return interaction.output_text
def answer_from_context_with_history(
    question,
    retrieved_chunks,
    chat_history,
):
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

    context_parts = []

    for chunk in retrieved_chunks:
        page_number = chunk["page_number"]
        text = chunk["text"]

        context_parts.append(
            f"Page {page_number}:\n{text}"
        )

    context = "\n\n".join(context_parts)

    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            history_text += f"User: {content}\n\n"
        elif role == "assistant":
            history_text += f"Assistant: {content}\n\n"

    prompt = f"""You are an AI document assistant.

Answer the user's latest question using only the document context below.

Use the recent conversation history only to understand follow-up questions.
Do not use information from the conversation history that is not supported by the document context.

If the answer is not in the document context, say:
"I could not find the answer in the provided document."

Recent conversation:
{history_text}

Document context:
{context}

Latest question:
{question}

Answer:
"""

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    return interaction.output_text