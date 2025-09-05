# src/sdoc_poc/retrieval_sdoc.py
import numpy as np
from typing import List, Tuple
from .store import StateDB
from .config import Config
from .azure_client import AzureClient

def sdoc_retrieve(db: StateDB, cfg: Config, client: AzureClient, doc_id: str, query: str, top_k: int = 5):
    qv = client.embed([query])[0]
    items = db.load_sdoc_units(doc_id)  # [(unit_hash, kind, text, idx, source_chunk_hash)]
    scored = []
    for uh, kind, text, idx, ref in items:
        v = db.load_embedding_vector(uh, cfg.embed_deployment)
        if v is None: 
            continue
        sim = float(np.dot(qv, v) / (np.linalg.norm(qv)*np.linalg.norm(v) + 1e-9))
        scored.append((sim, uh, idx, kind, text, ref))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]
