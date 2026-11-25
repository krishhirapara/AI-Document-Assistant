from pypdf import PdfReader
from text_splitter import split_pages_into_chunks
from rag_helper import create_vector_store, retrieve_relevant_chunks
from gemini_helper import answer_from_context

pdf_path = input("Enter the path to a small PDF: ")

reader = PdfReader(pdf_path)

pages = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    pages.append({
        "page_number": page_number,
        "text": text,
    })

chunks = split_pages_into_chunks(pages)

print(f"\nCreated {len(chunks)} chunks.")
print("Creating embeddings and saving to Chroma...")

collection = create_vector_store(chunks)

print("Vector store created successfully!")

question = input("\nEnter a question: ")

print("\nSearching for relevant chunks...")

results = retrieve_relevant_chunks(
    collection=collection,
    question=question,
)

print(f"\nTop {len(results)} relevant chunks:\n")

for index, chunk in enumerate(results, start=1):
    print(f"--- Chunk {index} | Page {chunk['page_number']} ---")
    print(chunk["text"][:500])
    print()

print("Generating answer with Gemini...")

answer = answer_from_context(
    question=question,
    retrieved_chunks=results,
)

print("\nGemini Answer:")
print(answer)