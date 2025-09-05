import os, numpy as np, sqlite3
from typing import List, Tuple
from .store import StateDB
from .azure_client import AzureClient
from .config import Config

def load_vec(db: StateDB, chunk_hash: str, model: str) -> np.ndarray:
    c = db.conn.cursor()
    c.execute("SELECT vec_path FROM embeddings WHERE chunk_hash=? AND model=?", (chunk_hash, model))
    row = c.fetchone()
    if not row: return None
    v = np.load(row[0]).astype('float32')
    return v

def doc_retrieve(db: StateDB, cfg: Config, client: AzureClient, doc_id: str, query: str, top_k: int = 5):
    # embed query
    qv = client.embed([query])[0]
    # iterate over doc chunks
    items = db.load_doc_chunks(doc_id)
    scored = []
    for chash, text, idx in items:
        v = load_vec(db, chash, cfg.embed_deployment)
        if v is None: continue
        sim = float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9))
        scored.append((sim, chash, idx, text))
    scored.sort(reverse=True, key=lambda t: t[0])
    return scored[:top_k]
