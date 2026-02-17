import argparse
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


BASE_DIR = Path(__file__).parent
DEFAULT_PDF_DIR = BASE_DIR / "data" / "pdfs"
DEFAULT_STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = DEFAULT_STORAGE_DIR / "chroma"
INDEX_DIR = DEFAULT_STORAGE_DIR / "index"
BM25_CORPUS_PATH = DEFAULT_STORAGE_DIR / "bm25_corpus.jsonl"


def load_environment() -> None:
    load_dotenv()


def ensure_directories() -> None:
    DEFAULT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def pdf_to_documents(pdf_path: Path) -> List[Document]:
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
    )
    documents: List[Document] = []
    for element in elements:
        text = getattr(element, "text", "")
        if not text:
            continue
        metadata = getattr(element, "metadata", None)
        page_number = getattr(metadata, "page_number", None) if metadata else None
        doc_metadata = {
            "document_name": pdf_path.name,
            "page_number": page_number,
            "element_type": type(element).__name__,
        }
        documents.append(Document(text=text, metadata=doc_metadata))
    return documents


def build_index_and_corpus(
    documents: List[Document],
    chroma_dir: Path,
    index_dir: Path,
    bm25_corpus_path: Path,
) -> None:
    if not documents:
        return
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.embed_model = embed_model

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection("finance_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(index_dir),
    )

    VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist(persist_dir=str(index_dir))

    bm25_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_corpus_path.open("w", encoding="utf-8") as f:
        for node in nodes:
            record = {
                "id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def ingest_files(
    pdf_paths: List[Path],
    chroma_dir: Path = CHROMA_DIR,
    index_dir: Path = INDEX_DIR,
    bm25_corpus_path: Path = BM25_CORPUS_PATH,
) -> None:
    load_environment()
    ensure_directories()

    all_documents: List[Document] = []
    for pdf_path in pdf_paths:
        if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
            all_documents.extend(pdf_to_documents(pdf_path))

    build_index_and_corpus(
        documents=all_documents,
        chroma_dir=chroma_dir,
        index_dir=index_dir,
        bm25_corpus_path=bm25_corpus_path,
    )


def ingest_pdf_directory(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    chroma_dir: Path = CHROMA_DIR,
    index_dir: Path = INDEX_DIR,
    bm25_corpus_path: Path = BM25_CORPUS_PATH,
) -> None:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    ingest_files(
        pdf_paths=pdf_paths,
        chroma_dir=chroma_dir,
        index_dir=index_dir,
        bm25_corpus_path=bm25_corpus_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=str(DEFAULT_PDF_DIR),
    )
    parser.add_argument(
        "--chroma_dir",
        type=str,
        default=str(CHROMA_DIR),
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str(INDEX_DIR),
    )
    parser.add_argument(
        "--bm25_corpus",
        type=str,
        default=str(BM25_CORPUS_PATH),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    chroma_dir = Path(args.chroma_dir)
    index_dir = Path(args.index_dir)
    bm25_corpus_path = Path(args.bm25_corpus)
    ingest_pdf_directory(
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        index_dir=index_dir,
        bm25_corpus_path=bm25_corpus_path,
    )


if __name__ == "__main__":
    main()

