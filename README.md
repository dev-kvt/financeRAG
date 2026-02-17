# Finance RAG Assistant

Production-grade Retrieval-Augmented Generation (RAG) system for financial documents (10-Ks, annual reports, etc.) built with:

- LlamaIndex for RAG orchestration
- ChromaDB for local vector storage
- OpenAI `text-embedding-3-small` for embeddings
- GPT-4o for financial analysis and generation
- Unstructured for high-fidelity PDF parsing (including tables)
- Rank-BM25 for keyword search
- Streamlit for the user interface

---

## Features

- Upload and ingest financial PDFs (10-Ks, annual reports)
- High-fidelity parsing of PDFs with table-aware extraction
- Semantic chunking with overlap to preserve context around figures
- Hybrid retrieval:
  - Semantic search over vector embeddings
  - Keyword search using BM25
- Financial-grade answer generation with:
  - Strict grounding to retrieved context
  - Explicit fallback when information is missing
  - Citations including document name and page number

---

## Project Structure

- `app.py` – Streamlit UI for uploading PDFs and asking questions
- `ingest.py` – Batch and programmatic ingestion of PDFs
- `engine.py` – Hybrid retrieval and financial QA engine
- `requirements.txt` – Python dependencies
- `.env` – Environment variables (API keys, endpoints)
- `data/pdfs/` – PDF storage directory (created at runtime)
- `storage/` – Persistent index and search data:
  - `storage/chroma/` – ChromaDB vector store
  - `storage/index/` – LlamaIndex index state
  - `storage/bm25_corpus.jsonl` – BM25 corpus with metadata

---

## Setup

### 1. Python environment

```bash
cd /Users/divyanshsingh/Developer/rag_exp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file at the project root (already scaffolded):

```text
OPENAI_API_KEY="your-openai-api-key"
OPENAI_API_BASE="https://api.openai.com/v1"
LLAMA_PARSE_API_KEY=""
```

Only `OPENAI_API_KEY` is required for the current setup. The other values are placeholders for advanced configurations.

---

## Ingesting PDFs

You can ingest PDFs in two ways: via the CLI or the Streamlit UI.

### Option A: CLI ingestion

1. Place your PDFs under:

   ```bash
   mkdir -p data/pdfs
   cp /path/to/your/*.pdf data/pdfs/
   ```

2. Run ingestion:

   ```bash
   python ingest.py --pdf_dir data/pdfs
   ```

This will:

- Parse PDFs using Unstructured with table-aware parsing
- Chunk documents with semantic chunking and 200-token overlap
- Populate the ChromaDB vector store (`storage/chroma/`)
- Persist LlamaIndex index state (`storage/index/`)
- Build the BM25 corpus (`storage/bm25_corpus.jsonl`)

### Option B: Ingestion via Streamlit

The Streamlit app also supports ingestion directly from the browser (see next section).

---

## Running the Streamlit App

1. Ensure the virtual environment is activated and dependencies are installed.

2. Start the app:

   ```bash
   streamlit run app.py
   ```

3. In the browser UI:

   - Use the sidebar to upload one or more financial PDFs.
   - Click **"Ingest uploaded PDFs"** to process and index them.
   - After ingestion completes, enter a question in the main panel.
   - Click **"Get answer"** to run the RAG pipeline.

The app will display:

- A grounded answer based on the ingested documents
- A list of sources, including:
  - Label (`Source 1`, `Source 2`, etc.)
  - Document name
  - Page number
  - Short text preview from the retrieved chunk

---

## Retrieval and QA Design

### High-fidelity parsing

`ingest.py` uses Unstructured's `partition_pdf` with:

- `strategy="hi_res"`
- `infer_table_structure=True`

This preserves table structure and metadata such as page numbers wherever possible.

### Chunking strategy

The ingestion pipeline uses `SentenceSplitter` from LlamaIndex with:

- Chunk size of 1024 tokens
- Overlap of 200 tokens

This maintains continuity across chunks so that related figures and narrative remain in context.

### Hybrid search

`engine.py` combines:

- Semantic search (LlamaIndex + ChromaDB using OpenAI embeddings)
- Keyword search (Rank-BM25 over the same chunk corpus)

Results from both retrievers are merged with a simple rank-fusion approach and then passed as context to the LLM.

### Financial validation and safety

The LLM (GPT-4o) is configured with a system prompt that instructs it to:

- Use only the provided context from financial filings and reports
- Remain objective and avoid speculation
- Preserve original numerical formatting (currencies, percentages, etc.)
- Prefer tabular data when both tables and narrative are available
- Respond exactly with:

  > Insufficient information in the provided documents to answer this question accurately.

  when the context does not support a high-confidence answer.

---

## Customization

- **Parser**: Swap Unstructured with LlamaParse if you prefer a different PDF parser. The `.env` already includes `LLAMA_PARSE_API_KEY` as a placeholder.
- **Vector DB**: The current setup uses ChromaDB. You can switch to a remote vector store (e.g., Pinecone) by updating the LlamaIndex vector store configuration in `engine.py` and `ingest.py`.
- **Models**:
  - Embeddings: `text-embedding-3-small` is used by default; you can replace it with a different embedding model.
  - LLM: GPT-4o is used for generation. Adjust the model name or temperature in `engine.py` as needed.

---

## Troubleshooting

- If ingestion succeeds but queries return no answers:
  - Confirm that PDFs were successfully saved under `data/pdfs/`.
  - Check that `storage/chroma/`, `storage/index/`, and `storage/bm25_corpus.jsonl` were created.
- If the app reports authentication errors:
  - Verify that `OPENAI_API_KEY` in `.env` is set and the environment is reloaded.

