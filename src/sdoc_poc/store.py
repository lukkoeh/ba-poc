import os, sqlite3, json, hashlib, numpy as np
from typing import List, Tuple, Optional
from .utils import file_sha256, now_ts

class StateDB:
    def __init__(self, path: str = "artifacts/state.db"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Allow cross-thread/process usage and reduce lock errors
        self.conn = sqlite3.connect(path, timeout=60.0, check_same_thread=False)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS documents(
            doc_id TEXT PRIMARY KEY,
            transcript_path TEXT NOT NULL,
            meta_path TEXT NOT NULL,
            transcript_hash TEXT NOT NULL,
            language TEXT,
            processed_at INTEGER
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            doc_id TEXT,
            idx INTEGER,
            chunk_hash TEXT PRIMARY KEY,
            text TEXT NOT NULL
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings(
            chunk_hash TEXT,
            model TEXT,
            dim INTEGER,
            vec_path TEXT,
            PRIMARY KEY (chunk_hash, model)
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS questions(
            q_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            question TEXT
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS judgments(
            q_id INTEGER,
            chunk_hash TEXT,
            judge_model TEXT,
            verdict TEXT,
            score REAL,
            rationale TEXT,
            PRIMARY KEY(q_id, chunk_hash, judge_model)
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS generated_meta(
        doc_id TEXT PRIMARY KEY,
        json TEXT NOT NULL
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS sdoc_units(
        doc_id TEXT,
        idx INTEGER,
        unit_hash TEXT PRIMARY KEY,
        kind TEXT,
        text TEXT NOT NULL,
        span_start INTEGER,
        span_end INTEGER,
        source_chunk_hash TEXT
        );
        """)

        self.conn.commit()

    def upsert_document(self, doc_id: str, transcript_path: str, meta_path: str, transcript_hash: str):
        c = self.conn.cursor()
        c.execute("""
        INSERT INTO documents(doc_id, transcript_path, meta_path, transcript_hash, processed_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(doc_id) DO UPDATE SET transcript_path=excluded.transcript_path, meta_path=excluded.meta_path, transcript_hash=excluded.transcript_hash
        """, (doc_id, transcript_path, meta_path, transcript_hash, now_ts()))
        self.conn.commit()

    def set_language(self, doc_id: str, lang: str):
        c = self.conn.cursor()
        c.execute("UPDATE documents SET language=?, processed_at=? WHERE doc_id=?", (lang, now_ts(), doc_id))
        self.conn.commit()

    def add_chunks(self, doc_id: str, chunks: List[str]):
        c = self.conn.cursor()
        for i, ch in enumerate(chunks):
            chash = hashlib.sha256((doc_id + f"#{i}" + ch).encode('utf-8')).hexdigest()
            c.execute("""
            INSERT OR IGNORE INTO chunks(doc_id, idx, chunk_hash, text) VALUES(?,?,?,?)
            """, (doc_id, i, chash, ch))
        self.conn.commit()

    def missing_embeddings(self, model: str) -> List[Tuple[str,str]]:
        c = self.conn.cursor()
        c.execute("""
        SELECT chunk_hash, text FROM chunks
        WHERE chunk_hash NOT IN (SELECT chunk_hash FROM embeddings WHERE model=?)
        ORDER BY doc_id, idx
        """, (model,))
        return c.fetchall()

    def record_embeddings(self, model: str, dim: int, pairs: List[Tuple[str, str, str]]):
        # pairs: (chunk_hash, vec_path, dim)
        c = self.conn.cursor()
        for chash, path, _ in pairs:
            c.execute("""
            INSERT OR REPLACE INTO embeddings(chunk_hash, model, dim, vec_path) VALUES(?,?,?,?)
            """, (chash, model, dim, path))
        self.conn.commit()

    def load_doc_chunks(self, doc_id: str) -> List[Tuple[str, str, int]]:
        c = self.conn.cursor()
        c.execute("SELECT chunk_hash, text, idx FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))
        return c.fetchall()

    def list_docs(self, limit: int = 0) -> List[str]:
        c = self.conn.cursor()
        q = "SELECT doc_id FROM documents ORDER BY doc_id"
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        c.execute(q)
        return [r[0] for r in c.fetchall()]

    def doc_paths(self, doc_id: str) -> Tuple[str, str]:
        c = self.conn.cursor()
        c.execute("SELECT transcript_path, meta_path FROM documents WHERE doc_id=?", (doc_id,))
        row = c.fetchone()
        return (row[0], row[1]) if row else (None, None)

    def add_questions(self, doc_id: str, questions: List[str]):
        c = self.conn.cursor()
        for q in questions:
            c.execute("INSERT INTO questions(doc_id, question) VALUES(?,?)", (doc_id, q))
        self.conn.commit()

    def doc_questions(self, doc_id: str) -> List[Tuple[int, str]]:
        c = self.conn.cursor()
        c.execute("SELECT q_id, question FROM questions WHERE doc_id=?", (doc_id,))
        return c.fetchall()

    def record_judgment(self, q_id: int, chunk_hash: str, model: str, verdict: str, score: float, rationale: str):
        c = self.conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO judgments(q_id, chunk_hash, judge_model, verdict, score, rationale)
        VALUES(?,?,?,?,?,?)
        """, (q_id, chunk_hash, model, verdict, score, rationale[:4000]))
        self.conn.commit()
    def get_language(self, doc_id: str) -> str:
        c = self.conn.cursor()
        c.execute("SELECT language FROM documents WHERE doc_id=?", (doc_id,))
        r = c.fetchone()
        return r[0] if r and r[0] else None
    def upsert_generated_meta(self, doc_id: str, meta: dict):
        c = self.conn.cursor()
        c.execute("INSERT INTO generated_meta(doc_id, json) VALUES(?,?) "
                "ON CONFLICT(doc_id) DO UPDATE SET json=excluded.json",
                (doc_id, json.dumps(meta, ensure_ascii=False)))
        self.conn.commit()

    def get_generated_meta(self, doc_id: str) -> dict | None:
        c = self.conn.cursor()
        c.execute("SELECT json FROM generated_meta WHERE doc_id=?", (doc_id,))
        r = c.fetchone()
        return json.loads(r[0]) if r else None

    def add_sdoc_units(self, doc_id: str, units: list[dict]):
        c = self.conn.cursor()
        for u in units:
            c.execute("""
                INSERT OR REPLACE INTO sdoc_units(doc_id, idx, unit_hash, kind, text, span_start, span_end, source_chunk_hash)
                VALUES(?,?,?,?,?,?,?,?)
            """, (doc_id, int(u["idx"]), u["unit_hash"], u["kind"], u["text"], int(u["span_start"]), int(u["span_end"]), u.get("source_chunk_hash")))
        self.conn.commit()

    def load_sdoc_units(self, doc_id: str):
        c = self.conn.cursor()
        c.execute("""SELECT unit_hash, kind, text, idx, source_chunk_hash
                    FROM sdoc_units WHERE doc_id=? ORDER BY idx""", (doc_id,))
        return c.fetchall()

    def missing_embeddings_for_sdoc_units(self, model: str):
        c = self.conn.cursor()
        c.execute("""
        SELECT unit_hash, text FROM sdoc_units
        WHERE unit_hash NOT IN (SELECT chunk_hash FROM embeddings WHERE model=?)
        ORDER BY doc_id, idx
        """, (model,))
        return c.fetchall()

    def chunk_hash_by_index(self, doc_id: str, idx: int) -> str | None:
        c = self.conn.cursor()
        c.execute("SELECT chunk_hash FROM chunks WHERE doc_id=? AND idx=?", (doc_id, idx))
        r = c.fetchone()
        return r[0] if r else None

    def get_chunk_text_by_hash(self, chash: str) -> str | None:
        c = self.conn.cursor()
        c.execute("SELECT text FROM chunks WHERE chunk_hash=?", (chash,))
        r = c.fetchone()
        return r[0] if r else None

    def load_embedding_vector(self, hash_id: str, model: str):
        c = self.conn.cursor()
        c.execute("SELECT vec_path FROM embeddings WHERE chunk_hash=? AND model=?", (hash_id, model))
        r = c.fetchone()
        if not r: return None
        import numpy as np
        return np.load(r[0]).astype("float32")
