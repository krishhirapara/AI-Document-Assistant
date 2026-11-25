import os
import tempfile


import streamlit as st
from pypdf import PdfReader


from text_splitter import split_pages_into_chunks
from rag_helper import create_vector_store, retrieve_relevant_chunks
from gemini_helper import answer_from_context_with_history



st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)



st.title("📄 AI Document Assistant")
st.caption(
    "Upload a PDF and ask questions. "
    "Answers are generated using relevant content retrieved from the document."
)



if "collection" not in st.session_state:
    st.session_state.collection = None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "document_name" not in st.session_state:
    st.session_state.document_name = None



with st.sidebar:
    st.header("Upload document")


    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )


    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")


        if st.button("Process PDF", type="primary"):
            with st.spinner("Extracting, chunking, embedding, and indexing PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_pdf_path = temp_file.name


                    reader = PdfReader(temp_pdf_path)


                    pages = []


                    for page_number, page in enumerate(
                        reader.pages,
                        start=1,
                    ):
                        text = page.extract_text() or ""


                        pages.append({
                            "page_number": page_number,
                            "text": text,
                        })


                    chunks = split_pages_into_chunks(pages)


                    if not chunks:
                        st.error(
                            "No readable text was found in this PDF. "
                            "It may be a scanned document."
                        )
                    else:
                        collection = create_vector_store(chunks)


                        st.session_state.collection = collection
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.chat_history = []


                        st.success(
                            f"Processed {len(chunks)} chunks from "
                            f"{len(pages)} pages."
                        )


                    os.unlink(temp_pdf_path)


                except Exception as error:
                    st.error(f"Could not process the PDF: {error}")


    if st.session_state.document_name:
        st.divider()
        st.subheader("Current document")
        st.write(st.session_state.document_name)


    st.divider()
    st.caption(
        "This assistant answers using the uploaded PDF. "
        "If the answer is not in the document, it will say so."
    )



st.header("Ask your document")


if st.session_state.collection is None:
    st.info(
        "Upload and process a PDF first. "
        "Then ask questions about its contents."
    )
else:
    question = st.chat_input(
        "Ask a question about the uploaded PDF..."
    )


    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })


        with st.spinner("Searching the document and generating an answer..."):
            try:
                retrieved_chunks = retrieve_relevant_chunks(
                    collection=st.session_state.collection,
                    question=question,
                )


                answer = answer_from_context_with_history(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    chat_history=st.session_state.chat_history,
             )


                source_pages = sorted(
                    list({
                        chunk["page_number"]
                        for chunk in retrieved_chunks
                        if chunk["page_number"] is not None
                    })
                )


                if source_pages:
                    answer_with_sources = (
                        f"{answer}\n\n"
                        f"**Source pages:** {', '.join(map(str, source_pages))}"
                    )
                else:
                    answer_with_sources = answer


                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_with_sources,
                    "sources": retrieved_chunks,
                })


            except Exception as error:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I could not generate an answer: {error}",
                    "sources": [],
                })


    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View retrieved sources"):
                    for index, source in enumerate(
                        message["sources"],
                        start=1,
                    ):
                        page_number = source.get("page_number", "Unknown")
                        text = source.get("text", "")


                        st.markdown(
                            f"**Source {index} — Page {page_number}**"
                        )
                        st.text(text[:400])


                        if index < len(message["sources"]):
                            st.divider()