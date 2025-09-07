# ---------------------------- pip Installations ----------------------------
# pip install -U transformers==4.44.2 accelerate==0.34.2 sentence-transformers==2.7.0 chromadb
# pip install "huggingface_hub[hf-transfer]"


# ---------------------------- Libraries & Environment ----------------------------
import chromadb
import os
import re
from sentence_transformers import SentenceTransformer
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


# Ensure we're allowed to fetch on first run (remove hard-offline flags if set)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
# Text-only: skip torchvision
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
# Silence Windows symlink warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# In PowerShell (one-time):  setx HF_HOME "C:\\hf_cache"
CACHE_DIR = os.environ.get("HF_HOME", None)


# ---------------------------- (5) Vector Store ----------------------------
VECTOR_STORE_PATH = r"C:\Users\Laptop\Desktop\Coding\LLM\Personal Projects\LLM RAG Pipeline\VectorDB\edgar_10k_chroma_20250906_222002"
client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

# See what's inside
collections = client.list_collections()
# print("[collections]", [c.name for c in collections])

# Get the collection
collection = client.get_collection("edgar_10k")
# print("Loaded collection:", collection.name, "with", collection.count(), "items")

# Uses the HF cache (CACHE_DIR) so it won’t re-download after the first time to speed up the script.
enc = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder=CACHE_DIR)


# ---------------------------- (6) LLM (CPU-friendly) ----------------------------
# Smaller model that fits in 8 GB RAM
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   
REVISION = "main"
# CPU dtype
DTYPE = torch.float32  

def load_llm():
    """
    Try local cache first (fast). If not present, fetch once to populate cache. To speed up
    later runs.
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, use_fast=True, trust_remote_code=True,
            local_files_only=True, cache_dir=CACHE_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True,
            torch_dtype=DTYPE, device_map="cpu", low_cpu_mem_usage=True,
            local_files_only=True, cache_dir=CACHE_DIR
        )
        return tok, model
    except Exception:
        print("Cache not ready yet; fetching from hub once to populate cache...")
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, use_fast=True, trust_remote_code=True,
            local_files_only=False, cache_dir=CACHE_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True,
            torch_dtype=DTYPE, device_map="cpu", low_cpu_mem_usage=True,
            local_files_only=False, cache_dir=CACHE_DIR
        )
        return tok, model

tok, model = load_llm()
print("✅ LLM loaded")

# Peek at some docs
# print(collection.get(where={"year": {"$gte": 2022}}, include=["metadatas","documents"], limit=3))
# print(collection.get(where={"item": {"$eq": "item_1"}}, include=["metadatas","documents"], limit=3))  
# print(collection.get(where={"company": {"$eq": "C.H. Robinson"}}, include=["metadatas","documents"], limit=3))  


# ---------------------------- Utilities ----------------------------
def _slug(s: str) -> str:
    """Normalize a company string for robust equality checks."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def company_candidates(name: str) -> list[str]:
    """
    Generate a handful of reasonable metadata variants.
    Handles case, punctuation, whitespace, 'Inc/Inc.', with/without comma, etc.
    """
    if not name:
        return []
    base = name.strip()
    variants = set()

    # Raw + case variants
    variants.update([base, base.upper(), base.title()])

    # Normalize 'Inc' endings
    inc_norm = re.sub(r"\binc\.?$", "Inc.", base, flags=re.I)
    inc_upper = re.sub(r"\binc\.?$", "INC.", base, flags=re.I)
    variants.update([inc_norm, inc_upper])

    # Add comma before Inc (e.g., "Company, Inc.")
    def with_comma(s: str) -> list[str]:
        return [
            re.sub(r"\s+Inc\.?$", ", Inc.", s, flags=re.I),
            re.sub(r"\s+INC\.?$", ", INC.", s, flags=re.I),
        ]
    for v in list(variants):
        variants.update(with_comma(v))

    # Remove commas variants and collapse whitespace
    variants.update({re.sub(",", "", v) for v in variants})
    variants = {re.sub(r"\s+", " ", v).strip() for v in variants}

    return list(variants)


NAME_TOKEN   = r"(?:[A-Z][A-Za-z&.\-']*|[A-Z])"   # allow single-letter tokens like C, H in the names of the company
PROPER_CHUNK = rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN})+"  # 2+ tokens

def _extract_company_verbatim(question: str) -> str | None:
    q = (question or "").strip()

    # pattern: "... for <Company> ?"
    m = re.search(rf"\bfor\s+({PROPER_CHUNK})(?:\W|$)", q)
    if m:
        return m.group(1).strip(" ,.")

    # pattern: "what was/were/is/are ... <Company> ..." and your original “does/did …”
    m = re.search(
        rf"(?i:\b(?:what\s+(?:was|were|is|are)\s+|what\s+does\s+(?:the\s+company\s+)?|did|does|do|when\s+did|how\s+many\s+did)\s*)"
        rf"({PROPER_CHUNK})"
        rf"(?i:\s+(?:have|report|state|say|disclose|announce|file|list|employ|help|prevent|post|generate|make)\b)?",
        q
    )
    if m:
        return m.group(1).strip(" ,.")

    # fallback: longest capitalized chunk
    cands = re.findall(rf"({PROPER_CHUNK})", q)
    if not cands:
        return None
    def score(s: str):
        s2 = s.lower()
        has_suffix = any(t in s2 for t in (" inc", " inc.", " corporation", " corp", " plc", " llc", " ltd"))
        return (has_suffix, len(s))
    return sorted({c.strip(" ,.") for c in cands}, key=score, reverse=True)[0]


def _infer_year_window(question: str):
    m = re.search(r"\b(19|20)\d{2}\b", question)
    if not m:
        return None
    y = int(m.group(0))
    return (y - 1, y + 1)


def _and_filter(clauses):
    """Return a Chroma `where` that is valid for 0/1/≥2 clauses."""
    clauses = [c for c in clauses if c]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def peek_docs(n: int = 10, where: dict | None = None):
    """Print n documents with company, date, item, and a short text preview."""
    res = collection.get(where=where, include=["metadatas", "documents"], limit=n)
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        company = meta.get("company", "?")
        date    = meta.get("date", "?")
        item    = meta.get("item", "?")
        preview = (doc or "").replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:240] + " …"
        print(f"{i}. {company} • {date} • {item}\n   {preview}\n")


# ---------------------------- Retrieval ----------------------------
def _resolve_company_in_db(company: str, year_rng=None) -> str | None:
    if not company:
        return None

    tries = [
        company,
        re.sub(r"\.?$", ".", company),
        re.sub(r"\s+Inc\.?$", ", Inc.", company, flags=re.I),
        re.sub(r"\s+INC\.?$", ", INC.", company, flags=re.I),
        company.title(),
    ]
    for t in dict.fromkeys(s.strip() for s in tries if s.strip()):
        try:
            r = collection.get(where={"company": {"$eq": t}}, include=["metadatas"], limit=1)
            if r.get("metadatas"):
                return t
        except Exception:
            pass

    # targeted scan using year window if available; otherwise a wider scan
    scan_where = None
    if year_rng:
        scan_where = {"$and": [{"year": {"$gte": year_rng[0]}}, {"year": {"$lte": year_rng[1]}}]}
    try:
        r = collection.get(where=scan_where, include=["metadatas"], limit=8000)
        for m in (r.get("metadatas") or []):
            c = (m or {}).get("company", "")
            if _slug(c) == _slug(company):
                return c
    except Exception:
        pass
    return None


def retrieve(question: str, k: int = 6):
    # ----- Build query embedding -----
    extra = ("employees headcount workforce 10-K Item 1"
             if re.search(r"\b(employee|headcount|workforce|staff|FTE|full[- ]time)\b", question, re.I)
             else "business description products services 10-K Item 1")
    q_text = f"query: {extra} — {question}"
    q_vec = enc.encode([q_text], normalize_embeddings=True)[0]

    # ----- Parse company/year -----
    company = _extract_company_verbatim(question)
    if company:
        company = re.sub(r"\b(?:have|report|state|say|disclose|announce|file|list|employ|help|prevent)\b\.?\s*$",
                         "", company, flags=re.I).strip(" ,")
    print(f"☑️  company extracted:                                  {company!r}")

    asked_slug = _slug(company) if company else None
    year_rng = _infer_year_window(question)

    # Helper: rerank + pack
    def _rank_and_pack(docs, metas):
        # strict company filter (after we already prefiltered by $eq)
        if asked_slug:
            pair = [(d, m) for d, m in zip(docs, metas) if _slug(m.get("company","")) == asked_slug]
            if pair:
                docs, metas = zip(*pair)
            else:
                return None

        embs = enc.encode(list(docs), normalize_embeddings=True)
        scores = [sum(a*b for a, b in zip(q_vec, v)) for v in embs]

        if re.search(r"\b(employee|headcount|workforce|staff|FTE|full[- ]time)\b", question, re.I):
            scores = [s + (0.05 if "employee" in (d or "").lower() else 0.0)
                      for s, d in zip(scores, docs)]

        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out_docs  = [docs[i] for i in order]
        out_metas = [metas[i] for i in order]

        ctx_blocks, debug = [], []
        for i, (doc, meta) in enumerate(zip(out_docs, out_metas), start=1):
            if len(doc) > 1500: doc = doc[:1500] + " …"
            label = f"[{i}] {meta.get('company','?')} • {meta.get('date','?')} • {meta.get('item','?')}"
            ctx_blocks.append(f"{label}\n{doc}")
            debug.append(label)
        print("☑️   Retrieved:                                          ", " | ".join(debug))

        return "\n\n".join(ctx_blocks)

    # ----- Stage 0: find the exact company string, then do $eq -----
    exact = _resolve_company_in_db(company) if company else None
    if exact:
        where = {"company": {"$eq": exact}}
        # Add numeric year window if we have one (string filters are touchy; numeric is safe)
        if year_rng:
            where = {"$and": [where, {"year": {"$gte": year_rng[0]}}, {"year": {"$lte": year_rng[1]}}]}
        try:
            print(f"☑️  prefilter via get company=$eq:                      {exact!r} -> where={where}")

            r = collection.get(where=where, include=["documents","metadatas"], limit=1200)
            docs  = r.get("documents") or []
            metas = r.get("metadatas") or []
            if docs and metas:
                packed = _rank_and_pack(docs, metas)
                if packed:
                    return packed
        except Exception as e:
            print(f"☑️  get-error company=$eq:                              {e}")


    # ----- Stage 1: numeric-only prefilter (broad), then slug filter -----
    def _year_where():
        if year_rng:
            return {"$and": [{"year": {"$gte": year_rng[0]}}, {"year": {"$lte": year_rng[1]}}]}
        return {"year": {"$gte": 2019}}

    for tag, where_filter, limit in [
        ("year-only", _year_where(), 800),
        ("no-filter", None, 1200),
    ]:
        try:
            print(f"☑️  prefilter via get:                                  {tag} -> where={where_filter}")

            r = collection.get(where=where_filter, include=["documents","metadatas"], limit=limit)
            docs  = r.get("documents") or []
            metas = r.get("metadatas") or []
            if not docs or not metas:
                continue
            # exact company slug check here
            if asked_slug:
                pair = [(d, m) for d, m in zip(docs, metas) if _slug(m.get("company","")) == asked_slug]
                if not pair:
                    continue
                docs, metas = zip(*pair)
            packed = _rank_and_pack(list(docs), list(metas))
            if packed:
                return packed
        except Exception as e:
            print(f"☑️  get-error {tag}:                                    {e}")



    # ----- Stage 2: broad semantic query (no where), then strict slug -----
    try:
        print("☑️  fallback pure semantic query, no where")
        r = collection.query(query_embeddings=[q_vec], n_results=600,
                             include=["documents","metadatas","distances"], where=None)
        if r.get("documents") and r["documents"][0]:
            docs  = list(r["documents"][0])
            metas = list(r["metadatas"][0])
            if asked_slug:
                pair = [(d, m) for d, m in zip(docs, metas) if _slug(m.get("company","")) == asked_slug]
                if not pair:
                    print(f"☑️  fallback semantic had no exact company match for:   {company!r}")

                    return f"No matching context found for '{company}'."
                docs, metas = zip(*pair)
            docs, metas = list(docs)[:k], list(metas)[:k]
            ctx_blocks, debug = [], []
            for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                if len(doc) > 1500: doc = doc[:1500] + " …"
                label = f"[{i}] {meta.get('company','?')} • {meta.get('date','?')} • {meta.get('item','?')}"
                ctx_blocks.append(f"{label}\n{doc}")
                debug.append(label)
            print(f"☑️  Retrieved (fallback):                               ", " | ".join(debug))

            return "\n\n".join(ctx_blocks)
    except Exception as e:
        print(f"☑️  fallback-error query():                             {e}")


    return f"No matching context found for '{company or 'the requested company'}'."



# ---------------------------- Answer ----------------------------
def answer(question: str, k: int = 3, max_new_tokens: int = 120):
    context = retrieve(question, k=k)

    # If retrieval failed or didn't match the requested company, don't generate.
    if isinstance(context, str) and context.lower().startswith("no matching context"):
        print(context)
        return

    messages = [
        {"role": "system", "content": "Answer strictly from the provided context. If the answer is not in context, say so."},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nCite sources using the bracket numbers like [1], [2]."}
    ]

    # Get prompt text from chat template, then tokenize with truncation
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Make sure pad token exists
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Live streaming to console
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,              # deterministic
        use_cache=False,              # RAM-friendly on 8 GB
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )

    print("🚀 generating...")
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for token in streamer:
        print(token, end="", flush=True)
    print("\n✅ generated")


# ---------------------------- Run ----------------------------
if __name__ == "__main__":
    # # Peek a few docs from recent years 
    # peek_docs(5, where={"year": {"$gte": 2022}})

    print("\n\n" + "="*80 + "\n")
    # Example RAG queries
    # answer("What does the company NortonLifeLock Inc help prevent?")
    # answer("In 2009, what was the consolidated total revenues for C H ROBINSON WORLDWIDE INC?")
    # Response: In 2009, the consolidated total revenues for C. H. Robinson Worldwide Inc. were $7.6 billion.


# system
# Answer strictly from the provided context. If the answer is not in context, say so.
# user
# Question: In 2009, what was the consolidated total revenues for C H ROBINSON WORLDWIDE INC?

# Context:
# [1] C H ROBINSON WORLDWIDE INC • 2010-03-01 • item_1
# Item 1. BUSINESS
# Overview
# C.H. Robinson Worldwide, Inc. (“C.H. Robinson,” “the company,” “we,” “us,” or “our”) is one of the largest third party logistics companies in the world with 2009 consolidated total revenues of $7.6 billion. We provide freight transportation services and logistics solutions to companies of all sizes, in a wide variety of industries. During 2009 we handled approximately 7.5 million shipments for more than 35,000 customers. We operate through a network of 235 offices, which we call branches, in North America, Europe, Asia, South America, Australia, and the Middle East. We have developed global multimodal transportation and distribution networks to provide logistics services worldwide. As a result, we have the capability of facilitating most aspects of the supply chain on behalf of our customers.
# We do not own the transportation equipment that is used to transport our customers’ freight. Through our contractual relationships with approximately 47,000 transportation companies, including motor carriers, railroads (primarily intermodal service providers), air freight and ocean carriers, we select and hire the appropriate transportation to meet our customers’ freight needs. Because we rely on subcontractors to transport freight, we can be flexible and focus on seeking solutions that work for our customers, rather than focusing on asset utilization. As an integral part of our transportation services, we provide a wide range of value-added logistics services,  …

# [2] C H ROBINSON WORLDWIDE INC • 2010-03-01 • item_7
# ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
# RESULTS OF OPERATIONS
# The following table illustrates our net revenue margins by services and products:
# The following table summarizes our net revenues by service line:
# The following table represents certain statements of operations data, shown as percentages of our net revenues:
# OVERVIEW
# Our company. We are a global provider of multimodal transportation services and logistics solutions, operating through a network of branch offices in North America, Europe, Asia, South America, Australia, and the Middle East. We do not own the transportation equipment that is used to transport our customers’ freight. We work with approximately 47,000 transportation companies worldwide, and through those relationships we select and hire the appropriate transportation providers to meet our customers’ needs. As an integral part of our transportation services, we provide a wide range of value added logistics services, such as supply chain analysis, freight consolidation, core carrier program management, and information reporting.
# In addition to multimodal transportation services, we also offer fresh produce sourcing and fee-based information services. Our Sourcing business is the buying, selling, and marketing of fresh produce. We purchase fresh produce through our network of produce suppliers and sell it to retail grocers and restaurant chains, produce wholesalers and foodservice providers. In many cas …

# [3] C H ROBINSON WORLDWIDE INC • 2010-03-01 • item_1
# Information Services is comprised of a C.H. Robinson subsidiary, T-Chek Systems, Inc., (T-Chek). T-Chek is a business-to-business provider of spend management and payment processing services. T-Chek’s customers are primarily motor carriers and truck stop chains. T-Chek’s platform supports open and closed loop networks that facilitate a variety of funds transfer, vendor payments, fuel purchasing, and online expense management.
# Our business model has been the main driver of our historical results and has positioned us for continued growth. One of our competitive advantages is our branch network of 235 offices, staffed by approximately 5,800 salespeople. These branch employees are in close proximity to both customers and transportation providers, which gives them broad knowledge of their local markets and enables them to respond quickly to customers’ and transportation providers’ changing needs. Branch employees act as a team in their sales efforts, customer service, and operations. Approximately 32 percent of our truckload shipments are shared transactions between branches. Our branches work together to complete transactions and collectively meet the needs of our customers. For large multi-location customers, we often coordinate our efforts in one branch and rely on multiple branch locations to deliver specific geographic or modal needs. Our methodology of providing services is very similar across all branches. Our North American branches have a common technology platform that  …

# Cite sources using the bracket numbers like [1], [2].
# assistant
# In 2009, the consolidated total revenues for C. H. Robinson Worldwide Inc. were $7.6 billion.
# ✅ generated
