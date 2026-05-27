"""Local RAG: ingestion + retrieval.

The knowledge base is just a folder of `.md`/`.txt` files. The ingestion
script writes chunks to `data/rag_chunks.json` and the retriever loads
that file at query time.

Retrieval uses TF-IDF cosine similarity when scikit-learn is available,
otherwise a keyword-overlap fallback. Both are deterministic and fast
enough for a portfolio MVP.

The whole pipeline is no-op-safe: when the knowledge base is empty,
`retrieve` returns []. The Evidence agent handles that case explicitly.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config as _config_mod
from .schemas import EvidenceChunk
from .utils import stable_hash, truncate


SUPPORTED_EXTENSIONS = (".md", ".txt")
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class IngestionResult:
    chunk_count: int
    files_processed: int
    output_path: str


SOURCE_TYPE_HINTS = {
    "current": "current_context",
    "context": "current_context",
    "news": "current_context",
    "history": "historical_analogy",
    "historical": "historical_analogy",
    "analogy": "historical_analogy",
    "strategy": "strategy_framework",
    "framework": "strategy_framework",
    "doctrine": "strategy_framework",
}


def _infer_source_type(path: str) -> str:
    lowered = path.lower()
    for key, value in SOURCE_TYPE_HINTS.items():
        if key in lowered:
            return value
    return "unknown"


def _infer_domain(path: str) -> str:
    parts = re.split(r"[\\/]", path.lower())
    for p in parts:
        if p in ("economy", "technology", "security", "strategy", "ideology", "historical"):
            return p
    base = os.path.basename(path).lower()
    if "econ" in base or "trade" in base or "chip" in base:
        return "economy"
    if "taiwan" in base or "security" in base or "military" in base:
        return "security"
    if "history" in base or "ussr" in base or "cold" in base:
        return "historical"
    if "ideolog" in base or "ccp" in base or "domestic" in base:
        return "ideology"
    return "general"


def _iter_source_files(root: str) -> Iterable[str]:
    if not os.path.isdir(root):
        return
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.startswith(".") or f.endswith("/.gitkeep"):
                continue
            if not f.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            yield os.path.join(dirpath, f)


def _chunk_text(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def ingest_knowledge_base(
    kb_dir: str = "knowledge_base",
    output_path: Optional[str] = None,
) -> IngestionResult:
    """Read all md/txt files, chunk them, and save to JSON.

    Safe to run when `kb_dir` is empty or missing - it just writes an
    empty chunks file so the retriever still has something to load.
    """
    output_path = output_path or _config_mod.CONFIG.rag_chunks_path
    chunks: List[Dict[str, Any]] = []
    files_processed = 0

    for path in _iter_source_files(kb_dir):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
        except OSError:
            continue
        files_processed += 1
        for idx, piece in enumerate(_chunk_text(raw)):
            chunks.append(
                {
                    "id": stable_hash(path, idx, piece[:64]),
                    "source_path": path,
                    "source_type": _infer_source_type(path),
                    "domain": _infer_domain(path),
                    "text": piece,
                }
            )

    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    return IngestionResult(
        chunk_count=len(chunks),
        files_processed=files_processed,
        output_path=output_path,
    )


def _load_chunks(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or _config_mod.CONFIG.rag_chunks_path
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _keyword_score(query_tokens: List[str], doc_text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    overlap = len(qset & dset)
    if overlap == 0:
        return 0.0
    return overlap / (len(qset) + 1e-9)


def _tfidf_scores(query: str, docs: List[str]) -> List[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return []
    if not docs:
        return []
    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    matrix = vec.fit_transform(docs + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    return [float(s) for s in sims]


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    chunks_path: Optional[str] = None,
) -> List[EvidenceChunk]:
    """Return the top-K most relevant chunks (empty list if KB is empty)."""
    top_k = top_k or _config_mod.CONFIG.max_retrieved_docs
    chunks = _load_chunks(chunks_path)
    if not chunks:
        return []

    docs = [c.get("text", "") for c in chunks]
    scores = _tfidf_scores(query, docs)
    if not scores:
        q_tokens = _tokenize(query)
        scores = [_keyword_score(q_tokens, d) for d in docs]

    ranked: List[Tuple[float, Dict[str, Any]]] = sorted(
        zip(scores, chunks), key=lambda x: x[0], reverse=True
    )
    out: List[EvidenceChunk] = []
    for score, ch in ranked[:top_k]:
        if score <= 0:
            continue
        out.append(
            EvidenceChunk(
                source_path=ch.get("source_path", ""),
                source_type=ch.get("source_type", "unknown"),
                domain=ch.get("domain", "general"),
                text=truncate(ch.get("text", ""), 800),
                score=float(score),
            )
        )
    return out


def retrieve_with_cache(
    query: str,
    scenario_mode: str,
    cache: Optional[Dict[str, List[EvidenceChunk]]] = None,
) -> Tuple[List[EvidenceChunk], bool]:
    """Wrapper that supports an in-memory retrieval cache per process.

    Returns (chunks, cache_hit).
    """
    if cache is None:
        cache = _RETRIEVAL_CACHE
    key = stable_hash(query, scenario_mode)
    if key in cache:
        return cache[key], True
    chunks = retrieve(query)
    cache[key] = chunks
    return chunks, False


_RETRIEVAL_CACHE: Dict[str, List[EvidenceChunk]] = {}
