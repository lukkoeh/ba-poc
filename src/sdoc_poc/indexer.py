import os, json, sqlite3
from typing import List, Dict
from .store import StateDB
from .config import Config
from .log import info
from .utils import dumps_json

def export_index(db: StateDB, cfg: Config):
    path = os.path.join(cfg.artifacts, "index.jsonl")
    c = db.conn.cursor()
    c.execute("""
        SELECT chunks.chunk_hash, chunks.text, chunks.idx, documents.doc_id
        FROM chunks JOIN documents ON chunks.doc_id = documents.doc_id
        ORDER BY documents.doc_id, chunks.idx
    """)
    with open(path, "w", encoding="utf-8") as f:
        for chash, text, idx, doc_id in c.fetchall():
            rec = {"chunk_hash": chash, "doc_id": doc_id, "idx": idx, "text": text}
            f.write(dumps_json(rec) + "\n")
    info(f"Index exportiert: {path}")
