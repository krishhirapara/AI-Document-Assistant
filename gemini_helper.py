import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"

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

    prompt = f"""You are a helpful AI document assistant.

    First decide whether the document context contains enough relevant information
    to answer the user's question.

    If the document context contains enough relevant information:
    - Answer using ONLY the document context.
    - Start the answer with: "Based on the provided document, "
    - Do not use outside knowledge.
    - On the final line, write exactly: ANSWER_SOURCE: DOCUMENT

    If the document context does not contain enough relevant information:
    - Use your general knowledge to answer helpfully.
    - Start the answer with exactly: "This is not included in the document, but here is the answer: "
    - Clearly explain that the answer is based on general knowledge, not the uploaded document.
    - Do not pretend that general-knowledge information came from the document.
    - On the final line, write exactly: ANSWER_SOURCE: GENERAL_KNOWLEDGE

    If the user asks you to explain something in simpler words, rephrase the relevant
    document information in easy-to-understand language.

    Recent conversation:
    {history_text}

    Document context:
    {context}

    User question:
    {question}

    Answer:
    """

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    response_text = interaction.output_text.strip()

    if "ANSWER_SOURCE: DOCUMENT" in response_text:
        answer = response_text.replace("ANSWER_SOURCE: DOCUMENT", "").strip()
        return answer, "document"

    if "ANSWER_SOURCE: GENERAL_KNOWLEDGE" in response_text:
        answer = response_text.replace(
            "ANSWER_SOURCE: GENERAL_KNOWLEDGE", ""
        ).strip()
        return answer, "general_knowledge"

    return response_text, "unknown"

def rewrite_question_for_retrieval(question, chat_history):
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

    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            history_text += f"User: {content}\n"
        elif role == "assistant":
            history_text += f"Assistant: {content}\n"

    prompt = f"""Rewrite the user's latest question as one standalone search query.

    The rewritten query must make sense without the conversation history.
    Keep the same language as the user's question.
    Do not answer the question.
    Return only the rewritten query. No explanation, no quotation marks, and no extra text.

    Recent conversation:
    {history_text}

    Latest question:
    {question}

    Rewritten search query:
    """

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    rewritten_question = interaction.output_text.strip()

    if not rewritten_question:
        return question

    return rewritten_question