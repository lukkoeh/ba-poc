import numpy as np
import os
from typing import List, Tuple
from .store import StateDB
from .azure_client import AzureClient
from .config import Config
from .log import info
from tqdm import tqdm

def ensure_embeddings_sdoc_units(db: StateDB, cfg: Config, client: AzureClient, batch_size: int = 32):
    missing = db.missing_embeddings_for_sdoc_units(cfg.embed_deployment)
    if not missing:
        info("Alle SDOC-Unit-Embeddings vorhanden.")
        return
    os.makedirs(os.path.join(cfg.artifacts, "vecs"), exist_ok=True)
    texts, hashes = [], []
    for h, t in missing:
        hashes.append(h); texts.append(t)
    for i in tqdm(range(0, len(texts), batch_size), desc="SDOC Embeddings"):
        batch = texts[i:i+batch_size]; bh = hashes[i:i+batch_size]
        vecs = client.embed(batch)
        pairs = []
        for h, v in zip(bh, vecs):
            path = os.path.join(cfg.artifacts, "vecs", f"{h}_{cfg.embed_deployment}.npy")
            np.save(path, v.astype("float32"))
            pairs.append((h, path, int(v.shape[0])))
        db.record_embeddings(cfg.embed_deployment, int(vecs.shape[1]), pairs)


def ensure_embeddings(db: StateDB, cfg: Config, client: AzureClient, batch_size: int = 32, dim: int = 1536):
    missing = db.missing_embeddings(cfg.embed_deployment)
    if not missing:
        info("Alle Embeddings bereits vorhanden.")
        return
    os.makedirs(os.path.join(cfg.artifacts, "vecs"), exist_ok=True)
    texts, hashes = [], []
    for chash, text in missing:
        hashes.append(chash); texts.append(text)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch = texts[i:i+batch_size]
        bhashes = hashes[i:i+batch_size]
        vecs = client.embed(batch)
        # dim from actual vector
        dim = int(vecs.shape[1])
        # save each vector
        pairs = []
        for h, v in zip(bhashes, vecs):
            path = os.path.join(cfg.artifacts, "vecs", f"{h}_{cfg.embed_deployment}.npy")
            np.save(path, v.astype("float32"))
            pairs.append((h, path, dim))
        db.record_embeddings(cfg.embed_deployment, dim, pairs)
