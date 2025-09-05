# src/sdoc_poc/analyze.py
from __future__ import annotations
import os, re, json, time, hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from .config import Config
from .store import StateDB
from .ingest import read_transcript
from .log import info, warn
from .lang_detect import detect_language

# --- einfache Satzsegmentierung (robust, ohne schwere Deps) ---
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> List[str]:
    s = [t.strip() for t in _SENT_SPLIT.split(text) if t.strip()]
    return s[:5000]  # hard cap

def _word_spans(text: str) -> List[Tuple[int,int,str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]

def _chunk_spans_by_words(text: str, max_words: int = 220, overlap_words: int = 30):
    spans = []
    tok = _word_spans(text)
    n = len(tok)
    i = 0
    while i < n:
        end = min(n, i + max_words)
        if n == 0: break
        cs, ce = tok[i][0], tok[end-1][1]
        chunk_text = " ".join(t[2] for t in tok[i:end])
        spans.append((i, end, cs, ce, chunk_text))
        if end == n: break
        i = max(0, end - overlap_words)
    return spans

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _llm_or_heuristic_summaries(cfg: Config, client, text: str, n_facts: int = 3) -> Dict[str, Any]:
    """
    Liefert {micro_summary, recap, key_facts[], sample_questions[]}; bei DRY_RUN heuristisch.
    """
    if not cfg.dry_run and getattr(client, "client", None) is not None:
        system = (
            "You analyze a transcript segment. Produce compact JSON with fields: "
            "micro_summary (<=30 words), recap (<=20 words), key_facts (3-5 single-sentence bullets), "
            "sample_questions (3-5 short, specific questions a user may ask). Return ONLY JSON."
        )
        user = text[:4000]
        try:
            resp = client.client.chat.completions.create(
                model=cfg.chat_deployment,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            # sanity
            data["key_facts"] = [str(x).strip() for x in (data.get("key_facts") or [])][:5]
            data["sample_questions"] = [str(x).strip("? ").strip() for x in (data.get("sample_questions") or [])][:5]
            data["micro_summary"] = str(data.get("micro_summary","")).strip()
            data["recap"] = str(data.get("recap","")).strip()
            return data
        except Exception:
            pass  # fall back to heuristic

    # Heuristik (DRY_RUN): nützlich & deterministisch
    sents = _split_sentences(text)
    micro = " ".join(sents[:1])[:240]
    recap = " ".join(sents[-1:])[:160]
    facts = [s for s in (sents[:2] + sents[-2:]) if len(s) > 20][:n_facts]
    qs = []
    for f in facts[:5]:
        # naive Frageform
        q = re.sub(r'^[A-ZÄÖÜ][a-zäöüß]+\s+','', f).strip()
        q = re.sub(r'\.$','', q)
        if q:
            qs.append(f"Worum geht es bei: {q}?")
    return {"micro_summary": micro, "recap": recap, "key_facts": facts, "sample_questions": qs[:5]}

def _llm_or_heuristic_entities_topics(cfg: Config, client, seg_text: str) -> Dict[str, Any]:
    """
    Liefert {"entities":[{"type":"PERSON|ORG|LOC","text":"...","span":[start,end]}], "topics":[...]}.
    Offsets sind relativ zum Segmenttext.
    """
    if not seg_text:
        return {"entities": [], "topics": []}

    # LLM-Weg (wenn nicht DRY_RUN)
    if not cfg.dry_run and getattr(client, "client", None) is not None:
        system = (
            "Extract PERSON, ORG, LOC entities with character spans [start,end] relative to the given TEXT. "
            "Also return 3-6 high-level topics (single words or short bigrams). "
            "Return ONLY JSON like: "
            "{\"entities\":[{\"type\":\"PERSON|ORG|LOC\",\"text\":\"...\",\"span\":[s,e]}],\"topics\":[\"...\"]}"
        )
        try:
            resp = client.client.chat.completions.create(
                model=cfg.chat_deployment,
                messages=[{"role":"system","content":system},{"role":"user","content":seg_text[:4000]}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            obj = json.loads(raw)
            ents = []
            for ent in (obj.get("entities") or []):
                t = str(ent.get("text","")).strip()
                typ = str(ent.get("type","")).upper()
                span = ent.get("span")
                if not t or typ not in ("PERSON","ORG","LOC"):
                    continue
                if not (isinstance(span,(list,tuple)) and len(span)==2 and isinstance(span[0],(int,float))):
                    # Fallback: erste Vorkommnis
                    s = seg_text.find(t)
                    if s >= 0:
                        span = [s, s+len(t)]
                    else:
                        continue
                ents.append({"type": typ, "text": t, "span": [int(span[0]), int(span[1])]})
            topics = [str(x).strip() for x in (obj.get("topics") or []) if str(x).strip()]
            return {"entities": ents[:20], "topics": topics[:6]}
        except Exception:
            pass  # Fallback: Heuristik

    # Heuristik (DRY_RUN): capitalized Phrasen + einfache Klassifikatoren
    stop = set("""a an and are as at be but by for if in into is it no not of on or s so such t that the their then there these they this to was were will with und der die das ein eine ist im in zu auf von mit für nicht oder den dem""".split())
    cand = []
    for m in re.finditer(r"(?:\b(?:[A-ZÄÖÜ][a-zäöüß]+)(?:\s+(?:[A-ZÄÖÜ][a-zäöüß]+|&|und|der|die|das|von|zu|de|la|le|di|del|the|of|and))*)", seg_text):
        txt = m.group(0).strip()
        if len(txt.split()) == 1 and txt.lower() in stop:
            continue
        typ = "ORG" if re.search(r"\b(AG|GmbH|SE|Inc\.?|LLC|Ltd\.?)\b", txt) else "PERSON"
        if re.search(r"\b(Berlin|München|Hamburg|Deutschland|Germany|Paris|London|New York)\b", txt):
            typ = "LOC"
        cand.append({"type": typ, "text": txt, "span": [m.start(), m.end()]})
    words = [w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß\-]{3,20}", seg_text)]
    words = [w for w in words if w not in stop]
    freq = Counter(words)
    topics = [w for w,_ in freq.most_common(6)]

    # Dedup nach (type,text)
    seen, ents = set(), []
    for e in cand:
        key = (e["type"], e["text"])
        if key in seen: 
            continue
        seen.add(key)
        ents.append(e)
    return {"entities": ents[:20], "topics": topics}

def analyze_and_generate_meta(db: StateDB, cfg: Config, client, doc_id: str) -> str:
    """
    Erzeugt Meta + SDOC-Units (Micro/Recap pro Segment) und persistiert:
      - generated_meta (DB + artifacts/meta/<doc_id>.generated.json)
      - sdoc_units (DB, inkl. Span/Chunk-Referenz)
    Gibt Pfad der geschriebenen Meta-JSON zurück.
    """
    # Quelldaten
    t_path, _ = db.doc_paths(doc_id)
    full = read_transcript(t_path) if t_path and os.path.exists(t_path) else ""
    if not full:
        warn(f"[analyze] Leerer Text für {doc_id}")
    lang = db.get_language(doc_id) or detect_language(full)

    # Segment-Spans synchron zur Chunk-Strategie
    spans = _chunk_spans_by_words(full, 220, 30)  # wie im Chunker

    # Section-Objekte + SDOC-Units
    sections: List[Dict[str,Any]] = []
    sdoc_units: List[Dict[str,Any]] = []
    doc_sample_questions: List[str] = []

    for idx, (ws,we,cs,ce,seg_text) in enumerate(spans):
        summ = _llm_or_heuristic_summaries(cfg, client, seg_text)
        sections.append({
            "idx": idx,
            "span_chars": [int(cs), int(ce)],
            "micro_summary": summ.get("micro_summary",""),
            "recap": summ.get("recap",""),
            "key_facts": summ.get("key_facts", []),
            "sample_questions": summ.get("sample_questions", []),
        })
        # Entities/Topics pro Segment extrahieren
        et = _llm_or_heuristic_entities_topics(cfg, client, seg_text)
        sections[-1]["entities"] = et.get("entities", [])

        # Doc-Topics einsammeln
        if "doc_topics_set" not in locals():
            doc_topics_set = set()
        for t in et.get("topics", []):
            if t: doc_topics_set.add(t)

        # Doc-weite Entities aggregieren (mit Section/Offset im Segment)
        if "doc_entities" not in locals():
            from collections import defaultdict
            doc_entities = defaultdict(lambda: {"type": None, "text": None, "count": 0, "occurrences": []})
        for e in sections[-1]["entities"]:
            key = (e["type"], e["text"])
            rec = doc_entities[key]
            rec["type"] = e["type"]; rec["text"] = e["text"]
            rec["count"] += 1
            rec["occurrences"].append({"section_idx": idx, "span": e["span"]})
        doc_sample_questions.extend(summ.get("sample_questions", []))

        # SDOC-Units pro Segment (Micro + Recap)
        base = f"{doc_id}#{idx}"
        micro_text = summ.get("micro_summary","")
        recap_text = summ.get("recap","")
        if micro_text:
            uhash = _hash(base + "|micro|" + micro_text)
            sdoc_units.append({
                "idx": len(sdoc_units),
                "unit_hash": uhash,
                "kind": "micro_summary",
                "text": micro_text,
                "span_start": int(cs),
                "span_end": int(ce),
                "source_chunk_hash": db.chunk_hash_by_index(doc_id, idx),
            })
        if recap_text:
            uhash = _hash(base + "|recap|" + recap_text)
            sdoc_units.append({
                "idx": len(sdoc_units),
                "unit_hash": uhash,
                "kind": "recap",
                "text": recap_text,
                "span_start": int(cs),
                "span_end": int(ce),
                "source_chunk_hash": db.chunk_hash_by_index(doc_id, idx),
            })

    # leichte Doc-Header-Meta
    title = (full.strip().split("\n",1)[0] or "")[:120]
    topics = sorted(list(doc_topics_set))[:12] if "doc_topics_set" in locals() else []
    entities = list(doc_entities.values()) if "doc_entities" in locals() else []

    meta = {
        "doc_id": doc_id,
        "language": lang,
        "title": title,
        "topics": topics,
        "entities": entities,
        # BA-kompatible Struktur: plan.sections[*].sample_questions
        "plan": {
            "sections": [
                {"idx": s["idx"], "sample_questions": s["sample_questions"]} for s in sections
            ]
        },
        # Vollständigerer Blick für spätere Zwecke:
        "sections": sections,
        "generated_at": int(time.time()),
    }

    # persistieren
    db.upsert_generated_meta(doc_id, meta)
    out_dir = os.path.join(cfg.artifacts, "meta")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{doc_id}.generated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # SDOC-Units persistieren
    db.add_sdoc_units(doc_id, sdoc_units)

    info(f"[analyze] Meta+SDOC-Units erzeugt für {doc_id} -> {out_path} (units={len(sdoc_units)})")
    return out_path
