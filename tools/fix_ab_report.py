#!/usr/bin/env python3
import os, json, sys

ROOT = os.path.dirname(os.path.dirname(__file__))
ART = os.path.join(ROOT, "artifacts")
PJ = os.path.join(ART, "report_ab.json")
PM = os.path.join(ART, "report_ab.md")

def clamp01(x):
    try:
        x = float(x)
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x
    except Exception:
        return 0.0

def fix_metrics(d):
    m = d.get("metrics")
    if not isinstance(m, dict):
        return False
    chg = False
    for side in ("baseline","sdoc"):
        if side in m and isinstance(m[side], dict):
            for key in ("recall@k","ndcg@k"):
                if key in m[side] and m[side][key] is not None:
                    v = clamp01(m[side][key])
                    if v != m[side][key]:
                        m[side][key] = v; chg = True
    d["metrics"] = m
    return chg

def main():
    if not os.path.exists(PJ):
        print(f"Not found: {PJ}")
        sys.exit(1)
    with open(PJ, "r", encoding="utf-8") as f:
        stats = json.load(f)

    changed = fix_metrics(stats)
    if not changed:
        print("No changes needed.")
    else:
        with open(PJ, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print("report_ab.json normalized.")

    # MD neu rendern (wenn Projektmodul verfügbar)
    try:
        sys.path.insert(0, ROOT)
        from src.sdoc_poc.evaluator import write_report_ab
        write_report_ab(PJ, PM, stats)
        print("report_ab.md regenerated.")
    except Exception as e:
        print(f"Could not regenerate MD via evaluator.write_report_ab: {e}")
        # Falls Import fehlschlägt, MD bleibt unverändert.

if __name__ == "__main__":
    main()
