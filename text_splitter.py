from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )

    chunks = []

    for page in pages:
        page_text = page["text"]

        if not page_text.strip():
            continue

        text_chunks = splitter.split_text(page_text)

        for text in text_chunks:
            chunks.append({
                "text": text,
                "page_number": page["page_number"],
            })

    return chunks