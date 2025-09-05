# src/sdoc_poc/sdoc_exporter.py
from __future__ import annotations
import os, re, json, hashlib, time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .store import StateDB
from .config import Config
from .ingest import read_transcript, read_meta
from .log import info, warn

SDOC_VERSION = "1.0"

def _word_spans(text: str) -> List[Tuple[int, int, str]]:
    # Entspricht text.split() (Whitespace), aber mit Zeichen-Offsets
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]

def _chunk_spans_by_words(text: str, max_words: int = 220, overlap_words: int = 30) -> List[Tuple[int,int,int,int,str]]:
    """
    Liefert Chunk-Spans synchron zur Chunking-Logik (wortbasiert, Overlap):
    Rückgabe: (w_start, w_end, char_start, char_end, chunk_text_norm)
    """
    tok = _word_spans(text)
    n = len(tok)
    spans = []
    i = 0
    while i < n:
        end = min(n, i + max_words)
        if n == 0:
            break
        char_s = tok[i][0]
        char_e = tok[end - 1][1]
        words = [t[2] for t in tok[i:end]]
        chunk_text = " ".join(words)
        spans.append((i, end, char_s, char_e, chunk_text))
        if end == n:
            break
        i = max(0, end - overlap_words)
    return spans

def _load_doc_row(db: StateDB, doc_id: str) -> Optional[Tuple[str, str, str, str, Optional[int]]]:
    c = db.conn.cursor()
    c.execute(
        "SELECT language, transcript_path, meta_path, transcript_hash, processed_at FROM documents WHERE doc_id=?",
        (doc_id,),
    )
    return c.fetchone()

def _embedding_for_chunk(db: StateDB, model: str, chunk_hash: str) -> Optional[Tuple[int, str]]:
    c = db.conn.cursor()
    c.execute(
        "SELECT dim, vec_path FROM embeddings WHERE chunk_hash=? AND model=?",
        (chunk_hash, model),
    )
    row = c.fetchone()
    return (int(row[0]), row[1]) if row else None

def _doc_chunks_from_db(db: StateDB, doc_id: str) -> Dict[int, Dict[str, Any]]:
    """
    Map: idx -> {chunk_hash, text}
    """
    out: Dict[int, Dict[str, Any]] = {}
    for chash, text, idx in db.load_doc_chunks(doc_id):
        out[int(idx)] = {"chunk_hash": chash, "text": text}
    return out

def _questions_and_judgments(db: StateDB, doc_id: str) -> List[Dict[str, Any]]:
    c = db.conn.cursor()
    c.execute("SELECT q_id, question FROM questions WHERE doc_id=?", (doc_id,))
    rows = c.fetchall()
    out = []
    for q_id, q in rows:
        c.execute(
            "SELECT judge_model, chunk_hash, verdict, score, rationale FROM judgments WHERE q_id=?",
            (q_id,),
        )
        j = c.fetchone()
        out.append(
            {
                "q_id": int(q_id),
                "question": q,
                "judgment": None
                if j is None
                else {
                    "judge_model": j[0],
                    "top1_chunk_hash": j[1],
                    "verdict": j[2],
                    "score": float(j[3]) if j[3] is not None else None,
                    "rationale": j[4],
                },
            }
        )
    return out

def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def export_sdocs(
    db: StateDB,
    cfg: Config,
    include_vectors: bool = False,
    only_doc_id: Optional[str] = None,
) -> str:
    """
    Exportiert vollständige SDOCs nach artifacts/sdoc/.
    - include_vectors=True: Embedding-Vektoren inline speichern (groß!)
    - only_doc_id: nur dieses Dokument exportieren
    Rückgabe: Pfad zum Manifest.
    """
    out_dir = os.path.join(cfg.artifacts, "sdoc")
    _ensure_dir(out_dir)

    if only_doc_id:
        doc_ids = [only_doc_id]
    else:
        doc_ids = db.list_docs(limit=cfg.max_docs or 0)

    manifest: List[Dict[str, Any]] = []
    exported = 0

    for doc_id in doc_ids:
        row = _load_doc_row(db, doc_id)
        if not row:
            warn(f"Dokument nicht im State: {doc_id}")
            continue
        language, t_path, m_path, t_hash, processed_at = row
        try:
            full_text = read_transcript(t_path) if t_path and os.path.exists(t_path) else ""
        except Exception:
            full_text = ""
        try:
            meta = read_meta(m_path) if m_path and os.path.exists(m_path) else {}
        except Exception:
            meta = {}

        # Chunks aus DB (Quelle der Wahrheit für Hashes & Texte)
        db_chunks = _doc_chunks_from_db(db, doc_id)

        # Für Offsets die Chunk-Spans erneut berechnen (identische Wortlogik wie im Chunker)
        spans = _chunk_spans_by_words(full_text, max_words=220, overlap_words=30)

        segments: List[Dict[str, Any]] = []
        for idx, (w_s, w_e, c_s, c_e, chunk_text_norm) in enumerate(spans):
            # Falls im DB-State vorhanden, nutze die dortigen Texte & Hashes (sicherer)
            entry = db_chunks.get(idx)
            if entry:
                chash = entry["chunk_hash"]
                ctext = entry["text"]
            else:
                # Fallback (sollte selten passieren): Hash wie in store.add_chunks
                chash = hashlib.sha256((doc_id + f"#{idx}" + chunk_text_norm).encode("utf-8")).hexdigest()
                ctext = chunk_text_norm

            emb_rec = _embedding_for_chunk(db, cfg.embed_deployment, chash)
            emb_obj: Optional[Dict[str, Any]] = None
            if emb_rec:
                dim, vec_path = emb_rec
                emb_obj = {
                    "model": cfg.embed_deployment,
                    "dim": int(dim),
                    "ref": vec_path,
                }
                if include_vectors:
                    try:
                        vec = np.load(vec_path).astype("float32").tolist()
                        emb_obj["vector"] = vec
                    except Exception:
                        emb_obj["vector"] = None

            segments.append(
                {
                    "idx": idx,
                    "hash": chash,
                    "char_start": int(c_s),
                    "char_end": int(c_e),
                    "text": ctext,
                    "embedding": emb_obj,
                }
            )

        questions = _questions_and_judgments(db, doc_id)

        sdoc = {
            "sdoc_version": SDOC_VERSION,
            "doc_id": doc_id,
            "language": language,
            "exported_at": int(time.time()),
            "source": {
                "transcript_path": t_path,
                "meta_path": m_path,
                "transcript_sha256": t_hash,
                "processed_at": processed_at,
            },
            "meta": meta,  # vollständige Meta roh eingebettet
            "text": {
                "length_chars": len(full_text),
                "length_words": len(full_text.split()),
                "sha256": _sha256_str(full_text) if full_text else None,
                "full": full_text,  # explizit vollständig, damit einzeln einsehbar
            },
            "segments": segments,
            "evaluation": {
                "questions": questions,
            },
            "artifacts": {
                "index_path": os.path.join(cfg.artifacts, "index.jsonl"),
                "vecs_dir": os.path.join(cfg.artifacts, "vecs"),
                "embedding_model": cfg.embed_deployment,
            },
        }

        out_path = os.path.join(out_dir, f"{doc_id}.sdoc.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sdoc, f, ensure_ascii=False, indent=2)

        # Hash der SDOC-Datei ins Manifest
        try:
            with open(out_path, "rb") as rf:
                sdoc_sha = hashlib.sha256(rf.read()).hexdigest()
        except Exception:
            sdoc_sha = None

        manifest.append(
            {
                "doc_id": doc_id,
                "path": out_path,
                "sha256": sdoc_sha,
                "segments": len(segments),
                "has_vectors_inline": bool(include_vectors),
            }
        )
        exported += 1

    manifest_path = os.path.join(out_dir, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exported": exported,
                "model": cfg.embed_deployment,
                "vectors_inline": bool(include_vectors),
                "items": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    info(f"SDOC Export erstellt: {manifest_path}")
    return manifest_path