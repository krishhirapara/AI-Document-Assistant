import os

from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "document_chunks"


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to your .env file."
        )

    return genai.Client(
        api_key=api_key,
        http_options={"timeout": 90000},
    )


def create_vector_store(chunks):
    client = get_client()

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    try:
        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    texts = []
    ids = []
    metadatas = []
    embeddings = []

    for index, chunk in enumerate(chunks):
        chunk_id = f"chunk_{index}"
        text = chunk["text"]

        if not text.strip():
            continue

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        embedding = result.embeddings[0].values

        texts.append(text)
        ids.append(chunk_id)
        metadatas.append({
            "page_number": chunk["page_number"],
        })
        embeddings.append(embedding)

    if texts:
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection


def retrieve_relevant_chunks(collection, question, top_k=2):
    client = get_client()

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
    )

    question_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for text, metadata in zip(documents, metadatas):
        retrieved_chunks.append({
            "text": text,
            "page_number": metadata.get("page_number"),
        })

    return retrieved_chunks