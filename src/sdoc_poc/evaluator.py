from .retrieval import doc_retrieve
from .retrieval_sdoc import sdoc_retrieve
from .store import StateDB
from .azure_client import AzureClient
from .config import Config
from .log import info
import statistics, json, os
from typing import Dict, List
from .utils import dumps_json
from math import log2

def _fuse_blocks(blocks: List[str], max_chars: int = 2000) -> str:
    """Fügt mehrere Textblöcke zusammen und begrenzt die Gesamtlänge."""
    out, used = [], 0
    for i, b in enumerate(blocks):
        if not b:
            continue
        piece = (("\n\n---\n" if i > 0 else "") + b.strip())
        remain = max_chars - used
        if remain <= 0:
            break
        if len(piece) > remain:
            out.append(piece[:remain])
            used += remain
            break
        out.append(piece); used += len(piece)
    return "".join(out)

def _anchor_bucket(idx: int, n_chunks: int) -> str:
    if n_chunks <= 0 or idx is None or idx < 0:
        return "none"
    third = max(n_chunks / 3.0, 1.0)
    if idx < third: return "early"
    if idx < 2*third: return "middle"
    return "late"

def _clamp01(x) -> float:
    try:
        x = float(x)
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x
    except Exception:
        return 0.0

def _ndcg_at_k(binary_rels, k: int, ideal_count: int) -> float:
    # Binäre Relevanzen, DCG/IDCG mit log2, Ergebnis in [0,1]
    rels = [1 if r else 0 for r in list(binary_rels)[:k]]
    dcg = sum((r / log2(i+2)) for i, r in enumerate(rels))
    idcg = sum((1 / log2(i+2)) for i in range(min(int(ideal_count or 0), k)))
    return _clamp01((dcg / idcg) if idcg > 0 else 0.0)

def build_questions_from_meta(meta: dict) -> List[str]:
    out = []
    # BA-kompatible Quelle: plan.sections[*].sample_questions
    plan = (meta or {}).get("plan", {})
    sections = plan.get("sections", [])
    for sec in sections:
        for q in sec.get("sample_questions", []) or []:
            out.append(str(q))
    # Fallback: direkte sections[*].sample_questions
    if not out and meta:
        for sec in meta.get("sections", []) or []:
            for q in sec.get("sample_questions", []) or []:
                out.append(str(q))
    return list(dict.fromkeys([q.strip() for q in out if q and len(q.strip()) >= 5]))[:16]

def evaluate_doc_ab(db: StateDB, cfg: Config, client: AzureClient, doc_id: str, meta: dict, max_q: int = 8):
    K = 5
    FUSE_TOPN = 5

    qs = build_questions_from_meta(meta)[:max_q]
    if not qs:
        return None
    db.add_questions(doc_id, qs)

    # Chunk-Mapping & Gold (Frage -> Section-Indizes)
    rows = db.load_doc_chunks(doc_id)  # [(chunk_hash, text, idx)]
    hash_to_idx = {h: i for (h, _t, i) in rows}
    n_chunks = len(rows)

    gold_map = {}
    for sec in (meta.get("sections") or []):
        sidx = int(sec.get("idx", -1))
        for q in (sec.get("sample_questions") or []):
            if isinstance(q, str) and q.strip():
                gold_map.setdefault(q.strip(), set()).add(sidx)

    # Aggregatoren
    results = []
    base_anchor = {"early":0, "middle":0, "late":0}
    sdoc_anchor = {"early":0, "middle":0, "late":0}
    gold_qs = 0
    no_gold_qs = 0
    sum_base_recall = sum_sdoc_recall = 0.0
    sum_base_ndcg = sum_sdoc_ndcg = 0.0
    sum_overlap = 0.0

    for q_id, q in db.doc_questions(doc_id):
        # A) Baseline
        base_hits = doc_retrieve(db, cfg, client, doc_id, q, top_k=K)
        base_blocks = [h[3] for h in (base_hits[:FUSE_TOPN] or [])]
        base_ctx = _fuse_blocks(base_blocks, max_chars=2000)
        base_j = client.judge(q, base_ctx)
        base_top_idxs = [h[2] for h in base_hits] if base_hits else []
        base_top1_idx = base_top_idxs[0] if base_top_idxs else None
        if base_hits:
            db.record_judgment(
                q_id, base_hits[0][1],
                (cfg.chat_deployment if not cfg.dry_run else "heuristic") + "|baseline",
                base_j.get("verdict","NO"), float(base_j.get("score",0.0)), base_j.get("rationale","")
            )

        # B) SDOC
        sdoc_hits = sdoc_retrieve(db, cfg, client, doc_id, q, top_k=K)
        blocks = []
        unit_hash = None
        unit_sim = 0.0
        for i, (sim, uh, _u_idx, kind, utext, ref_chunk) in enumerate(sdoc_hits[:FUSE_TOPN] or []):
            if i == 0:
                unit_hash = uh
                unit_sim = sim
            ref_text = db.get_chunk_text_by_hash(ref_chunk) or ""
            blocks.append((utext + "\n\n--- SOURCE SEGMENT ---\n" + ref_text[:1000]).strip())
        sdoc_ctx = _fuse_blocks(blocks, max_chars=2200)
        sdoc_j = client.judge(q, sdoc_ctx)
        sdoc_top_ref_hashes = [hit[5] for hit in sdoc_hits] if sdoc_hits else []
        sdoc_top_ref_idxs = [hash_to_idx.get(h, -1) for h in sdoc_top_ref_hashes]
        sdoc_top1_idx = sdoc_top_ref_idxs[0] if sdoc_top_ref_idxs else None
        if unit_hash:
            db.record_judgment(
                q_id, unit_hash,
                (cfg.chat_deployment if not cfg.dry_run else "heuristic") + "|sdoc",
                sdoc_j.get("verdict","NO"), float(sdoc_j.get("score",0.0)), sdoc_j.get("rationale","")
            )

        # Anchor-Buckets
        base_bucket = _anchor_bucket(base_top1_idx, n_chunks)
        sdoc_bucket = _anchor_bucket(sdoc_top1_idx, n_chunks)
        if base_bucket in base_anchor: base_anchor[base_bucket] += 1
        if sdoc_bucket in sdoc_anchor: sdoc_anchor[sdoc_bucket] += 1

        # Klassische Metriken
        gold_idxs = set(gold_map.get(q.strip(), set()))
        if gold_idxs:
            gold_qs += 1
            base_rels = [1 if i in gold_idxs else 0 for i in base_top_idxs]
            sdoc_rels = [1 if i in gold_idxs else 0 for i in sdoc_top_ref_idxs]
            # Recall@K = (#relevante in TopK) / (#relevante)
            base_recall = _clamp01(sum(base_rels) / max(len(gold_idxs), 1))
            sdoc_recall = _clamp01(sum(sdoc_rels) / max(len(gold_idxs), 1))
            base_ndcg = _ndcg_at_k(base_rels, K, len(gold_idxs))
            sdoc_ndcg = _ndcg_at_k(sdoc_rels, K, len(gold_idxs))
            sum_base_recall += base_recall
            sum_sdoc_recall += sdoc_recall
            sum_base_ndcg += base_ndcg
            sum_sdoc_ndcg += sdoc_ndcg
        else:
            no_gold_qs += 1
            # Heuristik: Overlap der referenzierten Chunks in Top-K
            overlap = 0.0
            if base_top_idxs and sdoc_top_ref_idxs:
                bs = set([i for i in base_top_idxs if i is not None and i >= 0])
                ss = set([i for i in sdoc_top_ref_idxs if i is not None and i >= 0])
                if K > 0:
                    overlap = len(bs & ss) / float(min(K, len(bs), len(ss)) or 1)
            sum_overlap += overlap

        results.append({
            "question": q,
            "baseline": {"top1_sim": float(base_hits[0][0]) if base_hits else 0.0, **base_j},
            "sdoc":     {"top1_sim": float(sdoc_hits[0][0]) if sdoc_hits else 0.0, **sdoc_j},
            "anchors": {"baseline": base_bucket, "sdoc": sdoc_bucket}
        })

    # Aggregation (LLM-Judge wie gehabt)
    def agg(key):
        vals = [r[key]["score"] for r in results]
        yes = sum(1 for r in results if r[key].get("verdict") == "YES")
        return {"avg_score": (sum(vals)/len(vals)) if vals else 0.0, "yes_rate": yes/max(1,len(results))}
    base_a = agg("baseline"); sdoc_a = agg("sdoc")
    delta = {"avg_score": sdoc_a["avg_score"] - base_a["avg_score"],
             "yes_rate": sdoc_a["yes_rate"] - base_a["yes_rate"]}

    # Aggregation Metriken/Anker (pro Dokument)
    base_anchor_total = sum(base_anchor.values())
    sdoc_anchor_total = sum(sdoc_anchor.values())
    metrics = {
        "k": K,
        "gold_qs": gold_qs,
        "no_gold_qs": no_gold_qs,
        "baseline": {
            "recall@k": _clamp01((sum_base_recall / gold_qs) if gold_qs else 0.0),
            "ndcg@k":   _clamp01((sum_base_ndcg / gold_qs)   if gold_qs else 0.0),
            "anchors": {"counts": base_anchor, "total": base_anchor_total}
        },
        "sdoc": {
            "recall@k": _clamp01((sum_sdoc_recall / gold_qs) if gold_qs else 0.0),
            "ndcg@k":   _clamp01((sum_sdoc_ndcg / gold_qs)   if gold_qs else 0.0),
            "anchors": {"counts": sdoc_anchor, "total": sdoc_anchor_total}
        },
        "overlap@k": (sum_overlap / no_gold_qs) if no_gold_qs else None
    }

    return {
        "doc_id": doc_id,
        "n_questions": len(results),
        "baseline": base_a,
        "sdoc": sdoc_a,
        "delta": delta,
        "metrics": metrics
    }

def write_report_ab(path_json: str, path_md: str, stats: Dict):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    lines = [
        "# SDOC A/B Report",
        "",
        f"Dokumente: {stats.get('n_docs',0)} – Fragen: {stats.get('n_questions',0)}",
        f"Baseline avg: {stats.get('baseline',{}).get('avg_score',0):.3f} | YES: {stats.get('baseline',{}).get('yes_rate',0):.2%}",
        f"SDOC     avg: {stats.get('sdoc',{}).get('avg_score',0):.3f} | YES: {stats.get('sdoc',{}).get('yes_rate',0):.2%}",
        f"Δ avg: {stats.get('delta',{}).get('avg_score',0):+.3f} | Δ YES: {stats.get('delta',{}).get('yes_rate',0):+.2%}",
        ""
    ]

    m = stats.get("metrics", {})
    if m:
        k = m.get("k", 5)
        # Klassische Metriken
        if m.get("gold_qs", 0):
            lines += [
                f"**Klassische Metriken (nur Fragen mit Gold, k={k}):**",
                f"- Recall@{k}:  base={m.get('baseline',{}).get('recall@k',0):.3f}  |  sdoc={m.get('sdoc',{}).get('recall@k',0):.3f}",
                f"- nDCG@{k}:    base={m.get('baseline',{}).get('ndcg@k',0):.3f}  |  sdoc={m.get('sdoc',{}).get('ndcg@k',0):.3f}",
                ""
            ]
        if m.get("overlap@k") is not None:
            lines += [f"**Heuristik (kein Gold):** Overlap@{k} = {m.get('overlap@k',0):.3f}", ""]

        # Anchor Coverage
        def _pct(anchor):
            tot = max(anchor.get("total",0), 1)
            c = anchor.get("counts", {})
            return (100*c.get("early",0)/tot, 100*c.get("middle",0)/tot, 100*c.get("late",0)/tot)
        b_e,b_m,b_l = _pct(m.get("baseline",{}).get("anchors",{}))
        s_e,s_m,s_l = _pct(m.get("sdoc",{}).get("anchors",{}))
        lines += [
            "**Anchor Coverage (Top‑1 Positionen):**",
            f"- Baseline: early {b_e:.1f}% | middle {b_m:.1f}% | late {b_l:.1f}%",
            f"- SDOC:     early {s_e:.1f}% | middle {s_m:.1f}% | late {s_l:.1f}%",
            ""
        ]

    lines.append("## Pro Dokument")
    for d in stats.get("by_doc", []):
        lines.append(
            f"- **{d['doc_id']}** – Q={d['n_questions']} | "
            f"base avg={d['baseline']['avg_score']:.3f}, YES={d['baseline']['yes_rate']:.2%}  "
            f"→ sdoc avg={d['sdoc']['avg_score']:.3f}, YES={d['sdoc']['yes_rate']:.2%}  "
            f"Δavg={d['delta']['avg_score']:+.3f}, ΔYES={d['delta']['yes_rate']:+.2%}"
        )

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
