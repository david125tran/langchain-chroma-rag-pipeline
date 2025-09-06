# ---------------------------------------- Terminal installs ----------------------------------------
# python -m pip install datasets langchain langchain-community chromadb sentence-transformers
# pip install fastembed


# ---------------------------------------- Imports ----------------------------------------
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


from typing import List
from datetime import datetime
import os


# ---------------------------------------- Config ----------------------------------------
KNOWLEDGE_BASE = "jlohding/sp500-edgar-10k"
PERSIST_DIR = "chroma_10k"
COLLECTION = "edgar_10k"


# ~1k–1.5k tokens per chunk depending on text; adjust to taste
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

# Force CPU (no GPU) for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""  

# ---------------------------------------- Helpers ----------------------------------------
def row_to_docs(row) -> List[Document]:
    """
    Converts one dataset row (a 10-K filing) into chunked Documents with lean metadata.
    Keeps only non-empty item_* sections.
    """
    # Keep CIK as string (preserve any zero padding)
    cik_str = str(row["cik"])
    # YYYY-MM-DD from pandas/pyarrow timestamp
    date_str = str(row["date"])[:10]
    year = int(date_str[:4])
    # Use full date to avoid same-year collisions (10-K vs 10-K/A, etc.)
    doc_id = f"{cik_str}-{date_str}"

    # Only include non-empty item_* sections
    sections = {
        k: v for k, v in row.items()
        if k.startswith("item_") and isinstance(v, str) and v.strip()
    }

    docs: List[Document] = []
    for item_name, item_text in sections.items():
        for idx, chunk in enumerate(splitter.split_text(item_text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "cik": cik_str,
                        "company": row["company"],
                        "date": date_str,
                        "year": year,
                        "item": item_name.lower(),   # normalize
                        "chunk_idx": idx,
                        # Optional: helps trace back to the original dataset row
                        "row_idx": int(row.get("__index_level_0__", -1)),
                    },
                )
            )
    return docs

# ---------------------------------------- (1) Load dataset ----------------------------------------
dataset = load_dataset(KNOWLEDGE_BASE, split="train")


# ---------------------------------------- (2) Preprocess & chunk (build once) ----------------------------------------
raw_docs: List[Document] = []
for row in dataset:
    raw_docs.extend(row_to_docs(row))

print(f"Created {len(raw_docs)} documents from {len(dataset)} filings.")
# Created 958320 documents from 6282 filings.


# ---------------------------------------- (3) Embeddings & Vector store ----------------------------------------
emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectordb = Chroma(
    collection_name=COLLECTION,
    embedding_function=emb,
    persist_directory=PERSIST_DIR,
)

# ---------------------------------------- (4) Index all at once, then persist ----------------------------------------
# (Optional) Stable IDs so reruns don't create duplicates:
ids = [f"{d.metadata['doc_id']}|{d.metadata['item']}|{d.metadata['chunk_idx']}" for d in raw_docs]

vectordb.add_documents(raw_docs, ids=ids)
vectordb.persist()

print("Finished indexing and persisted to disk.")
print(f"Persisted DB to: {PERSIST_DIR}")


# ---------------------------------------- (5) Retrieval example ----------------------------------------
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40})
q = "What did Apple disclose in Item 1A about cybersecurity in 2020?"
results = retriever.get_relevant_documents(q)
for r in results:
    print(r.metadata, r.page_content)
