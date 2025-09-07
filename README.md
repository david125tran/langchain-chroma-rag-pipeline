# Generative AI, RAG System

An end-to-end **Retrieval-Augmented Generation (RAG)** playground around SEC 10‑K filings. Built to play around with **RAG** concepts.

## Parts
- **Google Colab: Vector Store Builder** — chunk + embed 10-Ks and write a persistent **ChromaDB** collection (`edgar_10k`) using `BAAI/bge-small-en-v1.5`. Built on a GPU runtime (A100) for speed.
- **Download Vector Store** — copy the Chroma folder from Colab to your local machine.
- **Local Machine: RAG Pipeline** — retrieve top chunks and answer with a small local LLM (`Qwen/Qwen2.5-0.5B-Instruct`) on CPU (my local machine that doesn't have GPU runtime).

## Repo Layout
```
.
root/
├── Create_VectorDB.ipynb
├── RAG Pipeline.py
└── VectorDB/
   └── edgar_10k_chroma_*/       # Unzip your Vector Store DB to this directory 
      └── edgar_10k_chroma_YYYYMMDD_HHMMSS/               
      └── chroma.sqlite3.sqlite3
```
## Data / Knowledge Base

- **Dataset:** [S&P 500 EDGAR 10-K (Hugging Face)](https://huggingface.co/datasets/jlohding/sp500-edgar-10k)
- **Provenance:** Original filings from SEC EDGAR; mirrored on Hugging Face for research use.


## Setup
**Python 3.12**
```bash
pip install -U transformers==4.44.2 accelerate==0.34.2 sentence-transformers==2.7.0 chromadb
pip install "huggingface_hub[hf-transfer]"
```

Optional - Run this in PowerShell to make the first run of `RAG Pipeline.py` faster.  This first run of it will be very slow as it loads the Vector Store to the LLM.  
```bash
# Windows (PowerShell)
setx HF_HOME "C:\hf_cache"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TRANSFORMERS_NO_TORCHVISION=1
```

## Build (Colab)
1. Open `Create_VectorDB.ipynb` on a GPU runtime for faster runtime.
2. Ingest your 10‑K text, chunk, embed (`bge-small-en-v1.5`), and write a **Chroma persistent** collection named `edgar_10k`.
3. Download the resulting Chroma folder (e.g., `VectorDB/edgar_10k_chroma_YYYYMMDD_HHMMSS`).

## Run (Local)
1. Set the path in `rag/RAG Pipeline.py`:
   ```python
   VECTOR_STORE_PATH = r"C:\...\VectorDB\edgar_10k_chroma_YYYYMMDD_HHMMSS"
   ```
2. Run:
   ```bash
   python "RAG Pipeline.py"
   ```
3. Try questions like:
   - “What does the company **NortonLifeLock Inc** help prevent?”
   - “In 2009, what was the consolidated total revenues for **C H ROBINSON WORLDWIDE INC**?”

## How It Works
- Builds a query embedding from the question (+ a few guiding keywords).
- Prefilters candidates (prefer **numeric fields** like `year`) via `collection.get(...)`.
- Re-ranks with BGE cosine similarity on the client.
- Generates a short answer with Qwen 0.5B; cites retrieved chunks.

## Limitations (by design)
- **Company parsing is heuristic** (regex); sensitive to spelling, punctuation, and renames.
- **String filters can be brittle** in some Chroma setups. Prefer `$eq` for exact strings and **numeric filters** (`year`) over `$in`. Post-filter in Python if needed.
- **Tiny LLM** (Qwen 0.5B) → basic reasoning only; answers strictly from context.
- **No alias/CIK resolver** and **no hybrid search** (no BM25). Pure semantic → can miss exact numbers/phrases.
- Windows paths + different Chroma versions can behave differently.

## Ideas
- I need to update this script to do a fuzzy match (`rapidfuzz`) for company name rather than using a regex heuristic search method.  
- Hybrid retrieval: BM25 → semantic re-rank.
- Finder questions: aggregate hits by `meta['company']` when no company is given.
- Swap in a stronger LLM or an API model for better answers.



