import os, json, jsonlines
from typing import Dict, List, Tuple, Optional
from .utils import extract_id

def discover_pairs(transcripts_dir: str, meta_dir: str):
    """
    Liefert Liste (doc_id, transcript_path, meta_path_or_empty).
    Vorher: only inner-join -> 0 Paare wenn meta/ leer.
    Jetzt:  left-join  -> alle Transkripte, Meta optional.
    """
    trans = {}
    for fn in os.listdir(transcripts_dir):
        if fn.startswith("."):
            continue
        pid = extract_id(fn)
        if not pid:
            continue
        trans[pid] = os.path.join(transcripts_dir, fn)

    # Meta-Map optional aufbauen
    meta = {}
    if os.path.isdir(meta_dir):
        for fn in os.listdir(meta_dir):
            if fn.startswith("."):
                continue
            pid = extract_id(fn)
            if not pid:
                continue
            meta[pid] = os.path.join(meta_dir, fn)

    # Left-Join: jedes Transkript wird zum Paar, meta kann fehlen ("")
    pairs = []
    for pid in sorted(trans.keys()):
        t_path = trans[pid]
        m_path = meta.get(pid, "")  # leer = keine externe Meta vorhanden
        pairs.append((pid, t_path, m_path))
    return pairs

def read_transcript(path: str) -> str:
    if path.endswith(".jsonl"):
        texts = []
        with jsonlines.open(path, "r") as reader:
            for obj in reader:
                # Try common keys
                for key in ("text","content","utterance","line"):
                    if key in obj:
                        texts.append(str(obj[key]))
                        break
        return "\n".join(texts)
    elif path.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return "\n".join(str(x.get("text","")) for x in obj if isinstance(x, dict))
            if isinstance(obj, dict):
                for key in ("text","content","body"):
                    if key in obj:
                        return str(obj[key])
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def read_meta(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

