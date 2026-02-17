from pathlib import Path

import streamlit as st

from engine import FinanceRAGEngine, CHROMA_DIR, INDEX_DIR, BM25_CORPUS_PATH
from ingest import ingest_files, DEFAULT_PDF_DIR


st.set_page_config(page_title="Finance RAG Assistant", layout="wide")

st.title("Finance RAG Assistant")
st.write(
    "Upload financial PDFs such as 10-Ks or annual reports, run ingestion, and "
    "then ask questions. Answers are grounded in the ingested documents and "
    "include explicit citations."
)


def get_engine() -> FinanceRAGEngine:
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = FinanceRAGEngine(
            chroma_dir=CHROMA_DIR,
            index_dir=INDEX_DIR,
            bm25_corpus_path=BM25_CORPUS_PATH,
        )
    return st.session_state.rag_engine


with st.sidebar:
    st.header("Document ingestion")
    uploaded_files = st.file_uploader(
        "Upload financial PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    ingest_clicked = st.button("Ingest uploaded PDFs")

    if ingest_clicked and uploaded_files:
        pdf_dir = DEFAULT_PDF_DIR
        pdf_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for uploaded in uploaded_files:
            save_path = pdf_dir / uploaded.name
            with save_path.open("wb") as f:
                f.write(uploaded.read())
            saved_paths.append(save_path)
        ingest_files(saved_paths)
        if "rag_engine" in st.session_state:
            del st.session_state["rag_engine"]
        st.success("Ingestion completed. You can now query the documents.")
    elif ingest_clicked and not uploaded_files:
        st.warning("Please upload at least one PDF before running ingestion.")


st.subheader("Ask a question")
question = st.text_area(
    "Enter your question about the ingested financial documents",
    height=120,
)
ask_clicked = st.button("Get answer")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question before requesting an answer.")
    else:
        try:
            engine = get_engine()
            with st.spinner("Retrieving context and generating answer..."):
                answer, sources = engine.answer(question)
            st.markdown("### Answer")
            st.write(answer)

            if sources:
                st.markdown("### Sources")
                for source in sources:
                    label = source.get("label")
                    document_name = source.get("document_name")
                    page_number = source.get("page_number")
                    preview = source.get("preview")
                    st.markdown(
                        f"**{label}** — {document_name}, page {page_number}"
                    )
                    st.code(preview)
            else:
                st.info("No sources available for this answer.")
        except Exception as exc:
            st.error(f"An error occurred while answering the question: {exc}")

