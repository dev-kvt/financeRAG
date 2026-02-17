import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

import chromadb
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


BASE_DIR = Path(__file__).parent
DEFAULT_STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = DEFAULT_STORAGE_DIR / "chroma"
INDEX_DIR = DEFAULT_STORAGE_DIR / "index"
BM25_CORPUS_PATH = DEFAULT_STORAGE_DIR / "bm25_corpus.jsonl"


@dataclass
class SourceChunk:
    document_name: str
    page_number: Any
    text: str


class FinanceRAGEngine:
    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        index_dir: Path = INDEX_DIR,
        bm25_corpus_path: Path = BM25_CORPUS_PATH,
    ) -> None:
        load_dotenv()
        self.client = OpenAI()

        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.embed_model = embed_model

        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = chroma_client.get_or_create_collection("finance_docs")
        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(
            persist_dir=str(index_dir),
            vector_store=vector_store,
        )
        index = load_index_from_storage(storage_context)
        self.vector_retriever = index.as_retriever(similarity_top_k=10)

        self.bm25_corpus_path = bm25_corpus_path
        (
            self.bm25_index,
            self.bm25_texts,
            self.bm25_metadata,
            self.bm25_ids,
        ) = self._load_bm25_index(bm25_corpus_path)

        self.system_prompt = (
            "You are a senior financial analyst. Answer using only the provided "
            "context from financial filings and reports. Remain objective and "
            "avoid speculation. If the context does not contain the required "
            "information, respond exactly with "
            "\"Insufficient information in the provided documents to answer this "
            "question accurately.\" Focus on numerical precision and always keep "
            "the original formatting of figures, currencies, and percentages. "
            "Prioritize information that clearly comes from tables or structured "
            "disclosures when both narrative and tabular data are present. "
            "Include document names and page numbers in your reasoning when "
            "referencing specific figures."
        )

    def _load_bm25_index(
        self,
        corpus_path: Path,
    ) -> Tuple[Optional[BM25Okapi], List[str], List[Dict[str, Any]], List[str]]:
        texts: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        ids: List[str] = []
        if not corpus_path.exists():
            return None, texts, metadata_list, ids
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids.append(record["id"])
                texts.append(record["text"])
                metadata_list.append(record.get("metadata", {}))
        if not texts:
            return None, texts, metadata_list, ids
        tokenized_corpus = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, texts, metadata_list, ids

    def _hybrid_retrieve(
        self,
        query: str,
        top_k_semantic: int = 8,
        top_k_bm25: int = 8,
    ) -> List[SourceChunk]:
        semantic_nodes = self.vector_retriever.retrieve(query)

        combined: Dict[str, Dict[str, Any]] = {}

        for rank, node in enumerate(semantic_nodes):
            node_id = node.node_id
            metadata = dict(node.metadata or {})
            text = node.text
            rrf_score = 1.0 / (60 + rank)
            entry = combined.get(node_id, {})
            entry_score = entry.get("score", 0.0) + rrf_score
            entry.update(
                {
                    "score": entry_score,
                    "text": text,
                    "metadata": metadata,
                }
            )
            combined[node_id] = entry

        if self.bm25_index is not None and self.bm25_ids:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k_bm25]

            for rank, idx in enumerate(ranked_indices):
                if idx >= len(self.bm25_ids):
                    continue
                bm25_id = self.bm25_ids[idx]
                text = self.bm25_texts[idx]
                metadata = self.bm25_metadata[idx]
                rrf_score = 1.0 / (60 + rank)
                entry = combined.get(bm25_id, {})
                entry_score = entry.get("score", 0.0) + rrf_score
                if "text" not in entry:
                    entry.update(
                        {
                            "text": text,
                            "metadata": metadata,
                        }
                    )
                entry["score"] = entry_score
                combined[bm25_id] = entry

        sorted_items = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        chunks: List[SourceChunk] = []
        for item in sorted_items[: max(top_k_semantic, top_k_bm25)]:
            metadata = item.get("metadata") or {}
            document_name = str(metadata.get("document_name", "unknown"))
            page_number = metadata.get("page_number")
            text = str(item.get("text", ""))
            chunks.append(
                SourceChunk(
                    document_name=document_name,
                    page_number=page_number,
                    text=text,
                )
            )
        return chunks

    def _build_context_block(
        self,
        chunks: List[SourceChunk],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            header = (
                f"[Source {idx}: {chunk.document_name}, page {chunk.page_number}]"
            )
            parts.append(header)
            parts.append(chunk.text)
            sources.append(
                {
                    "label": f"Source {idx}",
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "preview": chunk.text[:400],
                }
            )
        context_text = "\n\n".join(parts)
        return context_text, sources

    def answer(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        chunks = self._hybrid_retrieve(question)
        if not chunks:
            return (
                "Insufficient information in the provided documents to answer this "
                "question accurately.",
                [],
            )
        context_text, sources = self._build_context_block(chunks)
        user_content = (
            "You are given the following context from financial documents. "
            "Use only this context to answer the question.\n\n"
            f"{context_text}\n\n"
            "User question:\n"
            f"{question}\n\n"
            "If the context does not contain enough information to answer with high "
            "confidence, respond exactly with "
            "\"Insufficient information in the provided documents to answer this "
            "question accurately.\""
        )
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        answer_text = response.choices[0].message.content or ""
        return answer_text.strip(), sources
