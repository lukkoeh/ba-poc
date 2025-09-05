# src/sdoc_poc/augment_units.py
from __future__ import annotations
import os, json, re, hashlib
from typing import List, Dict, Any, Tuple

from .store import StateDB
from .config import Config
from .ingest import read_transcript
from .log import info, warn

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _heuristic_entities(seg_text: str) -> List[Dict[str, Any]]:
    # sehr leichte, deterministische Heuristik (keine LLM-Kosten)
    ents = []
    if not seg_text:
        return ents
    stop = set("""a an and are as at be but by for if in into is it no not of on or s so such t that the their then there these they this to was were will with und der die das ein eine ist im in zu auf von mit für nicht oder den dem""".split())
    for m in re.finditer(r"(?:\b(?:[A-ZÄÖÜ][a-zäöüß]+)(?:\s+(?:[A-ZÄÖÜ][a-zäöüß]+|&|und|der|die|das|von|zu|de|la|le|di|del|the|of|and))*)", seg_text):
        txt = m.group(0).strip()
        if len(txt.split()) == 1 and txt.lower() in stop:
            continue
        typ = "ORG" if re.search(r"\b(AG|GmbH|SE|Inc\.?|LLC|Ltd\.?)\b", txt) else "PERSON"
        if re.search(r"\b(Berlin|München|Hamburg|Deutschland|Germany|Paris|London|New York)\b", txt):
            typ = "LOC"
        ents.append({"type": typ, "text": txt, "span": [m.start(), m.end()]})
    # dedup
    seen, out = set(), []
    for e in ents:
        k = (e["type"], e["text"])
        if k in seen: 
            continue
        seen.add(k)
        out.append(e)
    return out[:20]

def augment_units_from_meta(db: StateDB, cfg: Config, client, doc_id: str, include_entities: bool = True, include_key_facts: bool = True) -> Dict[str, int]:
    """
    Erzeugt zusätzliche SDOC-Units (key_fact, entity) NUR aus vorhandenem Material.
    - Liest bevorzugt generated_meta aus DB; fällt zurück auf data/meta/<id>.json
    - Nutzt Chunk-Index -> Chunk-Hash aus dem StateDB
    - Schreibt neue Units in sdoc_units (INSERT OR REPLACE, Schlüssel: unit_hash)
    - Aktualisiert generated_meta in DB + artifacts/meta/<id>.generated.json mit Countern
    Rückgabe: {"added_key_facts": X, "added_entities": Y}
    """
    # Meta laden
    meta = db.get_generated_meta(doc_id)
    if not meta:
        _, meta_path = db.doc_paths(doc_id)
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        else:
            meta = {}
    sections = meta.get("sections") or []

    # Hilfs-Mapping: idx->(chunk_hash, text)
    chunk_rows = db.load_doc_chunks(doc_id)  # [(chunk_hash, text, idx)]
    idx_to_hash = {int(i): h for (h, _t, i) in chunk_rows}
    idx_to_text = {int(i): t for (h, t, i) in chunk_rows}

    # Existierende Units für fortlaufenden idx ermitteln
    existing = db.load_sdoc_units(doc_id)  # [(unit_hash, kind, text, idx, source_chunk_hash)]
    next_idx = len(existing)

    added_key, added_ent = 0, 0
    new_units = []

    for sec in sections:
        idx = int(sec.get("idx", -1))
        if idx < 0:
            continue
        chash = idx_to_hash.get(idx)
        seg_text = idx_to_text.get(idx, "")
        span = sec.get("span_chars") or [0, 0]
        cs, ce = int(span[0]), int(span[1])

        # key_facts -> Units
        if include_key_facts:
            for fact in (sec.get("key_facts") or []):
                txt = (str(fact) or "").strip()
                if not txt:
                    continue
                uhash = _hash(f"{doc_id}#{idx}|key|{txt}")
                new_units.append({
                    "idx": next_idx, "unit_hash": uhash, "kind": "key_fact",
                    "text": txt, "span_start": cs, "span_end": ce, "source_chunk_hash": chash
                })
                next_idx += 1
                added_key += 1

        # entities -> Units (falls nicht bereits vorhanden in meta: heuristisch erzeugen)
        if include_entities:
            ents = sec.get("entities")
            if ents is None:
                ents = _heuristic_entities(seg_text)
                sec["entities"] = ents  # in Meta nachziehen
            for e in (ents or []):
                etxt = (str(e.get("text","")) or "").strip()
                if not etxt:
                    continue
                uhash = _hash(f"{doc_id}#{idx}|ent|{e.get('type','UNK')}|{etxt}")
                new_units.append({
                    "idx": next_idx, "unit_hash": uhash, "kind": "entity",
                    "text": etxt, "span_start": cs, "span_end": ce, "source_chunk_hash": chash
                })
                next_idx += 1
                added_ent += 1

        # kleine Counter ins Meta schreiben (leichtgewichtige „Einfügung in Meta“)
        sec["key_fact_units_count"] = len(sec.get("key_facts") or [])
        sec["entity_units_count"] = len(sec.get("entities") or [])

    # persistieren (Units)
    if new_units:
        db.add_sdoc_units(doc_id, new_units)

    # Meta aktualisieren (DB + Datei)
    meta.setdefault("unit_summary", {})
    meta["unit_summary"]["added_key_fact_units"] = meta["unit_summary"].get("added_key_fact_units", 0) + added_key
    meta["unit_summary"]["added_entity_units"] = meta["unit_summary"].get("added_entity_units", 0) + added_ent
    db.upsert_generated_meta(doc_id, meta)

    # Datei schreiben (falls generated_meta-Datei existiert oder Artifacts-Verzeichnis verfügbar)
    out_dir = os.path.join(cfg.artifacts, "meta")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{doc_id}.generated.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        warn(f"[augment] Konnte Meta-Datei für {doc_id} nicht schreiben: {e}")

    info(f"[augment] {doc_id}: +key_facts={added_key}, +entities={added_ent}, units_total_added={len(new_units)}")
    return {"added_key_facts": added_key, "added_entities": added_ent}
