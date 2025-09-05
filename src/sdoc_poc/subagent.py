import os, ast, pathlib
from typing import List
from .azure_client import AzureClient
from .config import Config
from .log import info, warn

TEMPLATE = """You are a code quality reviewer. Review the following Python file for clarity, robustness, and correctness for an ETL+RAG pipeline PoC. 
Point out critical issues first, then quick wins. Be concise and practical.\n\nFILE: {path}\n\nCONTENT:\n{code}\n"""

def static_scan(py_paths: List[str]) -> List[str]:
    issues = []
    for p in py_paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                src = f.read()
            ast.parse(src)
        except SyntaxError as e:
            issues.append(f"SyntaxError in {p}: {e}")
        except Exception as e:
            issues.append(f"Fehler in {p}: {e}")
    return issues

def run_subagent(repo_root: str, cfg: Config, client: AzureClient):
    out_path = os.path.join(cfg.artifacts, "subagent_report.md")
    py_paths = [str(p) for p in pathlib.Path(repo_root).rglob("*.py") if "/.venv/" not in str(p)]
    issues = static_scan(py_paths)
    lines = ["# Subagent Report", "", "## Statischer Scan", *[f"- {i}" for i in issues or ["keine kritischen Syntaxfehler"]]]
    # Optional LLM review (one file sample to limit cost)
    if not cfg.dry_run and py_paths:
        sample = py_paths[0]
        with open(sample, 'r', encoding='utf-8') as f:
            code = f.read()[:16000]
        critique = client.judge("Review this code for robustness", code)
        lines += ["", "## LLM Review (Stichprobe)", f"- Verdict: {critique.get('verdict','')}", f"- Score: {critique.get('score',0)}", f"- Notes: {critique.get('rationale','')[:1000]}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
