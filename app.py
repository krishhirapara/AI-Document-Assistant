import streamlit as st
from pypdf import PdfReader
from text_splitter import split_pages_into_chunks
from gemini_helper import get_answer


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon=None,
    layout="centered",
)


st.title("AI Document Assistant")
st.write("Upload a PDF to extract and preview its text.")


uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)


if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)


        if reader.is_encrypted:
            st.warning(
                "This PDF is password-protected. "
                "Please use an unprotected PDF for now."
            )
        else:
            pages = []


            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""


                pages.append({
                    "page_number": page_number,
                    "text": text,
                })


            full_text = "\n\n".join(
                page["text"] for page in pages
            )


            if full_text.strip():
                st.success("PDF text extracted successfully!")


                st.write(f"File: {uploaded_file.name}")
                st.write(f"Pages: {len(pages)}")
                st.write(f"Characters extracted: {len(full_text):,}")


                with st.expander(
                    "View extracted text",
                    expanded=True,
                ):
                    st.text_area(
                        "Text preview",
                        value=full_text[:15000],
                        height=350,
                    )


                st.caption(
                    "The preview shows up to 15,000 characters."
                )


                chunks = split_pages_into_chunks(pages)


                st.subheader("Text chunks")
                st.write(f"Total chunks created: {len(chunks)}")


                for index, chunk in enumerate(chunks[:5], start=1):
                    with st.expander(
                        f"Chunk {index} — Page {chunk['page_number']}"
                    ):
                        st.write(chunk["text"])
                        st.caption(
                            f"Characters: {len(chunk['text'])}"
                        )


                st.caption(
                    "Showing up to the first 5 chunks."
                )


                st.subheader("Ask about this document")


                question = st.text_input(
                    "Enter your question",
                    placeholder="For example: What is this document about?",
                )

                use_general_knowledge = st.checkbox(
                    "Allow general knowledge answers",
                    value=False,
                )

                if st.button("Get answer"):
                    if not question.strip():
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Getting answer..."):
                            try:
                                answer = get_answer(
                                    question=question,
                                    document_text=full_text,
                                    use_general_knowledge=use_general_knowledge,
                                )


                                st.subheader("Answer")
                                st.write(answer)


                            except Exception as error:
                                st.error(
                                    "Could not get an answer from Gemini. "
                                    "Please try again."
                                )
                                st.caption(str(error))


            else:
                st.warning(
                    "No readable text was found. "
                    "Try a PDF where you can select and copy the text."
                )


    except Exception:
        st.error(
            "We could not read this PDF. "
            "Please try another valid, unprotected PDF."
        )
else:
    st.info("Upload a PDF to get started.")