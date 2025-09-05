import os, argparse, json
import time
import concurrent.futures as cf
from dotenv import load_dotenv
from src.sdoc_poc.log import info, warn, error, get_progress
from src.sdoc_poc.config import Config
from src.sdoc_poc.azure_client import AzureClient
from src.sdoc_poc.store import StateDB
from src.sdoc_poc.ingest import discover_pairs, read_transcript, read_meta
from src.sdoc_poc.lang_detect import detect_language
from src.sdoc_poc.chunker import chunk_text
from src.sdoc_poc.embeddings import ensure_embeddings, ensure_embeddings_sdoc_units
from src.sdoc_poc.indexer import export_index
from src.sdoc_poc.subagent import run_subagent
from src.sdoc_poc.sdoc_exporter import export_sdocs
from src.sdoc_poc.analyze import analyze_and_generate_meta
from src.sdoc_poc.evaluator import evaluate_doc_ab, write_report_ab
from src.sdoc_poc.utils import get_git_hash  # Snapshot
from src.sdoc_poc import __version__ as SDOC_VERSION
import platform, time
from src.sdoc_poc.augment_units import augment_units_from_meta


def step_augment(cfg: Config, db: StateDB, client: AzureClient):
    docs = db.list_docs(limit=cfg.max_docs or 0)
    for doc_id in docs:
        augment_units_from_meta(db, cfg, client, doc_id, include_entities=True, include_key_facts=True)
    # Nur neue/missing Unit-Embeddings rechnen:
    ensure_embeddings_sdoc_units(db, cfg, client)

def step_ingest(cfg: Config, db: StateDB, transcripts_dir: str, meta_dir: str):
    pairs = discover_pairs(transcripts_dir, meta_dir)
    if cfg.max_docs and cfg.max_docs > 0:
        pairs = pairs[:cfg.max_docs]
    info(f"Gefundene Paare: {len(pairs)}")
    with get_progress() as prog:
        task = prog.add_task("Ingest", total=len(pairs))
        for doc_id, t_path, m_path in pairs:
            t_hash = ""
            try:
                t_hash = __import__('src.sdoc_poc.utils', fromlist=['']).utils.file_sha256(t_path)
            except Exception:
                pass
            db.upsert_document(doc_id, t_path, m_path, t_hash)
            text = read_transcript(t_path)
            lang = detect_language(text)
            db.set_language(doc_id, lang)
            chunks = chunk_text(text)
            db.add_chunks(doc_id, chunks)
            prog.update(task, advance=1)

def step_embed(cfg: Config, db: StateDB, client: AzureClient):
    ensure_embeddings(db, cfg, client)
    ensure_embeddings_sdoc_units(db, cfg, client)

def step_index(cfg: Config, db: StateDB):
    export_index(db, cfg)

def _eval_doc_worker(doc_id: str, db_path: str):
    """Worker for parallel evaluation of a single document.
    Returns the per-doc result dict or None.
    """
    try:
        from src.sdoc_poc.config import Config
        from src.sdoc_poc.azure_client import AzureClient
        from src.sdoc_poc.store import StateDB
        from src.sdoc_poc.ingest import read_meta
        from src.sdoc_poc.evaluator import evaluate_doc_ab
        from src.sdoc_poc.log import info, warn

        cfg = Config()
        client = AzureClient(cfg)
        db = StateDB(db_path)

        info(f"[Worker {os.getpid()}] Eval start: {doc_id}")

        meta = db.get_generated_meta(doc_id)
        if not meta:
            _t, meta_path = db.doc_paths(doc_id)
            try:
                meta = read_meta(meta_path)
            except Exception:
                meta = {}

        res = evaluate_doc_ab(db, cfg, client, doc_id, meta, max_q=8)
        info(f"[Worker {os.getpid()}] Eval done: {doc_id}")
        return res
    except Exception as e:
        try:
            from src.sdoc_poc.log import error
            error(f"Eval worker failed for {doc_id}: {e}")
        except Exception:
            pass
        return None

def step_eval(cfg: Config, db: StateDB, client: AzureClient, parallel: int = 1):
    """
    Parallelisierte Evaluation über Dokumente.
    - parallel <= 1  -> sequentiell, schreibt Judgments wie bisher in die DB.
    - parallel > 1   -> multithreaded; nutzt pro Worker eine eigene DB-Verbindung & Azure-Client.
                        Judgments werden NICHT in die DB persistiert (Proxy), Ergebnisse fließen ins Report.
    """
    import os, traceback
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.sdoc_poc.ingest import read_meta
    from src.sdoc_poc.evaluator import evaluate_doc_ab, write_report_ab
    from src.sdoc_poc.azure_client import AzureClient
    from src.sdoc_poc.store import StateDB
    from src.sdoc_poc.log import get_progress, warn, info

    docs = db.list_docs(limit=cfg.max_docs or 0)
    n_docs = len(docs)
    if n_docs == 0:
        warn("Keine Dokumente für Eval gefunden."); return

    # --- Helfer: DB-Proxy für parallele Runs (verhindert Schreibkonflikte) ---
    class _EvalDBProxy:
        """Nur für parallel>1: Read-Delegation an eigene StateDB; add_questions/doc_questions/record_judgment lokal."""
        def __init__(self, path: str):
            self._inner = StateDB(path)
            self._qs = {}          # doc_id -> List[(qid, q)]
            self._jid = 0          # lokale Laufnummern für (nicht persistierte) q_ids

        # Schreib-APIs: lokal puffern / no-op
        def add_questions(self, doc_id: str, qs):
            # Erzeuge lokale q_ids (1..n) pro Dokument
            self._qs[doc_id] = [(i+1, q) for i, q in enumerate(qs or [])]

        def doc_questions(self, doc_id: str):
            return list(self._qs.get(doc_id, []))

        def record_judgment(self, *args, **kwargs):
            # no-op im Parallelmodus, um DB-Locks zu vermeiden
            return None

        # Alles andere an echte StateDB durchreichen (READS)
        def __getattr__(self, name):
            return getattr(self._inner, name)

    # --- Worker: evaluiert EIN Dokument, optional parallel ---
    def _eval_one(doc_id: str, use_proxy: bool):
        try:
            # lokale Ressourcen je Thread (thread-safe)
            local_client = AzureClient(cfg)
            if use_proxy:
                local_db = _EvalDBProxy(os.path.join(cfg.artifacts, "state.db"))
            else:
                # sequentiell -> Original-DB verwenden, damit Judgments persistieren
                local_db = db

            # Meta laden (bevorzugt generated_meta aus DB)
            meta = local_db.get_generated_meta(doc_id)
            if not meta:
                _t, meta_path = local_db.doc_paths(doc_id)
                try:
                    meta = read_meta(meta_path)
                except Exception:
                    meta = {}

            # eigentliche Doc-Eval (liefert A/B + Metriken + Anchors pro Doc)
            res = evaluate_doc_ab(local_db, cfg, local_client, doc_id, meta, max_q=8)
            return (doc_id, res, None)
        except Exception as e:
            return (doc_id, None, f"{type(e).__name__}: {e}")

    # --- Ausführung: parallel oder sequentiell ---
    by_doc = []
    total_q = 0

    # Aggregatoren (klassische Metriken/Anchors)
    gold_qs_total = 0
    no_gold_total = 0
    sum_base_recall = sum_sdoc_recall = 0.0
    sum_base_ndcg = sum_sdoc_ndcg = 0.0
    anchor_counts = {"baseline": {"early":0,"middle":0,"late":0}, "sdoc": {"early":0,"middle":0,"late":0}}
    anchor_totals = {"baseline":0, "sdoc":0}
    K_used = 5  # wird aus Ergebnissen übernommen, falls vorhanden

    use_proxy = True if (parallel and parallel > 1) else False

    with get_progress() as prog:
        task = prog.add_task("Eval", total=n_docs)

        if not use_proxy:
            # Sequentiell (persistiert Judgments wie gehabt)
            for doc_id in docs:
                _doc, res, err = _eval_one(doc_id, use_proxy=False)
                if err:
                    warn(f"[eval] {doc_id}: {err}")
                elif res:
                    by_doc.append(res); total_q += res.get("n_questions", 0)
                prog.update(task, advance=1)
        else:
            # Parallel (keine DB-Writes; schneller & robust gg. SQLite-Locks)
            workers = max(1, int(parallel))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_eval_one, doc_id, True): doc_id for doc_id in docs}
                for fut in as_completed(futs):
                    doc_id = futs[fut]
                    _doc, res, err = fut.result()
                    if err:
                        warn(f"[eval] {doc_id}: {err}")
                    elif res:
                        by_doc.append(res); total_q += res.get("n_questions", 0)
                    prog.update(task, advance=1)

    # --- Top-Level Aggregation (A/B + klassische Metriken + Anchors) ---
    if by_doc:
        base_avg = sum(d["baseline"]["avg_score"] for d in by_doc)/len(by_doc)
        base_yes = sum(d["baseline"]["yes_rate"] for d in by_doc)/len(by_doc)
        sdoc_avg = sum(d["sdoc"]["avg_score"] for d in by_doc)/len(by_doc)
        sdoc_yes = sum(d["sdoc"]["yes_rate"] for d in by_doc)/len(by_doc)
        delta_avg = sdoc_avg - base_avg
        delta_yes = sdoc_yes - base_yes
    else:
        base_avg = base_yes = sdoc_avg = sdoc_yes = delta_avg = delta_yes = 0.0

    # Klassische Metriken & Anchors einsammeln/mitteln (sofern in res vorhanden)
    for res in by_doc:
        m = res.get("metrics", {})
        if not m: 
            continue
        K_used = m.get("k", K_used)
        g = int(m.get("gold_qs", 0) or 0)
        n = int(m.get("no_gold_qs", 0) or 0)
        gold_qs_total += g
        no_gold_total += n
        if g:
            br = float(m.get("baseline",{}).get("recall@k") or 0.0)
            sr = float(m.get("sdoc",{}).get("recall@k") or 0.0)
            bn = float(m.get("baseline",{}).get("ndcg@k") or 0.0)
            sn = float(m.get("sdoc",{}).get("ndcg@k") or 0.0)
            sum_base_recall += br * g
            sum_sdoc_recall += sr * g
            sum_base_ndcg += bn * g
            sum_sdoc_ndcg += sn * g
        if n and (m.get("overlap@k") is not None):
            try:
                sum_overlap = float(m.get("overlap@k") or 0.0) * n
            except Exception:
                pass
        for side in ("baseline","sdoc"):
            a = m.get(side, {}).get("anchors", {})
            c = a.get("counts", {})
            anchor_counts[side]["early"]  += int(c.get("early",0)  or 0)
            anchor_counts[side]["middle"] += int(c.get("middle",0) or 0)
            anchor_counts[side]["late"]   += int(c.get("late",0)   or 0)
            anchor_totals[side] += int(a.get("total",0) or 0)

    metrics = {
        "k": K_used,
        "gold_qs": gold_qs_total,
        "baseline": {
            "recall@k": (sum_base_recall / gold_qs_total) if gold_qs_total else None,
            "ndcg@k":   (sum_base_ndcg / gold_qs_total)   if gold_qs_total else None,
            "anchors": {"counts": anchor_counts["baseline"], "total": anchor_totals["baseline"]}
        },
        "sdoc": {
            "recall@k": (sum_sdoc_recall / gold_qs_total) if gold_qs_total else None,
            "ndcg@k":   (sum_sdoc_ndcg / gold_qs_total)   if gold_qs_total else None,
            "anchors": {"counts": anchor_counts["sdoc"], "total": anchor_totals["sdoc"]}
        },
        "overlap@k": None  # gesamt optional – in evaluate_doc_ab wird pro Doc geführt
    }

    stats = {
        "n_docs": len(by_doc),
        "n_questions": total_q,
        "baseline": {"avg_score": base_avg, "yes_rate": base_yes},
        "sdoc": {"avg_score": sdoc_avg, "yes_rate": sdoc_yes},
        "delta": {"avg_score": delta_avg, "yes_rate": delta_yes},
        "metrics": metrics,
        "by_doc": by_doc,
    }
    write_report_ab(os.path.join(cfg.artifacts, "report_ab.json"),
                    os.path.join(cfg.artifacts, "report_ab.md"), stats)
    info("A/B Evaluation abgeschlossen -> artifacts/report_ab.md")


def step_subagent(cfg: Config, client: AzureClient):
    path = run_subagent(os.getcwd(), cfg, client)
    info(f"Subagent-Report: {path}")

def step_analyze(cfg: Config, db: StateDB, client: AzureClient):
    docs = db.list_docs(limit=cfg.max_docs or 0)
    for doc_id in docs:
        analyze_and_generate_meta(db, cfg, client, doc_id)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["all","ingest","analyze","augment","embed","index","eval","subagent","export"], help="Pipeline-Schritt oder all")
    parser.add_argument("--transcripts", dest="transcripts", default="data/transcripts", help="Pfad zu Transkripten")
    parser.add_argument("--meta", dest="meta", default="data/meta", help="Pfad zu Meta-Dateien")
    parser.add_argument("--doc", dest="doc", default=None, help="Nur dieses Dokument exportieren (doc_id).")
    parser.add_argument("--include-vectors", dest="include_vectors", action="store_true", help="Embedding-Vektoren inline in SDOC ablegen (groß).")
    parser.add_argument("--parallel", dest="parallel", type=int, default=1, help="Parallelität für eval (Prozesse).")
    args = parser.parse_args()

    cfg = Config()
    client = AzureClient(cfg)
    db = StateDB(os.path.join(cfg.artifacts, "state.db"))
    
    try:
        snap = {
            "timestamp": int(time.time()),
            "version": SDOC_VERSION,
            "git": {"hash": get_git_hash()},
            "azure": {
                "endpoint": cfg.endpoint,
                "api_version": cfg.api_version,
                "embed_deployment": cfg.embed_deployment,
                "chat_deployment": cfg.chat_deployment,
            },
            "dry_run": cfg.dry_run,
            "max_docs": cfg.max_docs,
            "python": platform.python_version(),
            "entry_command": args.command,
        }
        with open(os.path.join(cfg.artifacts, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
        info("Run-Fingerprint -> artifacts/run_config.json")
    except Exception as e:
        warn(f"Run-Fingerprint konnte nicht geschrieben werden: {e}")

    if args.command in ("ingest","all"): step_ingest(cfg, db, args.transcripts, args.meta)
    if args.command in ("analyze","all"): step_analyze(cfg, db, client)
    if args.command in ("augment", "all"): step_augment(cfg, db, client)
    if args.command in ("embed","all"):
        step_embed(cfg, db, client)              # Chunks
        ensure_embeddings_sdoc_units(db, cfg, client)  # SDOC-Units
    if args.command in ("index","all"): step_index(cfg, db)
    if args.command in ("eval","all"): step_eval(cfg, db, client, parallel=args.parallel)
    if args.command in ("subagent","all"): step_subagent(cfg, client)
    if args.command == "export": 
        out = export_sdocs(db, cfg, include_vectors=args.include_vectors, only_doc_id=args.doc)
        info(f"SDOC Export abgeschlossen -> {out}")
        return


if __name__ == "__main__":
    main()
