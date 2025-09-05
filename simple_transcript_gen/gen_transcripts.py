#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_transcripts.py — vereinfachter Generator für lange, informationsdichte Interview-Transkripte
mit Azure OpenAI (gpt-4o/gpt-5-chat). Ein einzelnes Skript, kein Packaging notwendig.

Neu (für BA-Setup):
- **Deterministische Sprachverteilung** über mehrere Transkripte (empfohlen: de=0.4, en=0.4, mix=0.2).
- **Token-bewusstes Stoppen**: Rollout läuft weiter, bis mind. target_output_tokens (≈ Completion) erreicht sind
  ODER max_turns überschritten würden.
- **Erweiterte fiktive Welt** (Company, 16+ Projekte, Artefakte, Policies) für viel Inhalt.
- **Mehrstufiger Rollout** (Steps × Turns) mit Ankern (Start/Mitte/Ende) gegen Position Bias.

Abhängigkeiten: openai>=1.40.0, python-dotenv>=1.0.1
"""

import os, sys, json, time, uuid, random, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from string import Template
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# ------------------ Fiktive Welt (erweitert) ------------------

WORLD = {
    "company": {
        "name": "Novereon Systems GmbH",
        "industry": "IT/Software",
        "mission": "Cloud-native Data & Platform Engineering for regulated industries",
        "values": ["Safety First", "Evidence over Hype", "Customer-Obsessed", "Sustainable Velocity"],
        "departments": ["Platform", "Data", "Security", "Mobile", "SRE", "FinOps", "QA", "Architecture", "UX"],
        "policies": [
            {"id":"POL-SEC-001","title":"Least Privilege & JIT Access"},
            {"id":"POL-QA-014","title":"Risk-Based Testing & Traceability"},
            {"id":"POL-FIN-007","title":"Cloud Cost Budgets & Quotas"}
        ],
        "conventions": {
            "ticket_prefixes": {"gateway":"GW","billing":"BIL","iam":"IAM","dr":"DR","obs":"OBS"},
            "rfc_prefix":"RFC","runbook_prefix":"RB","sla_prefix":"SLA","audit_prefix":"AUD"
        },
        "glossary": {
            "SLO":"Service Level Objective",
            "SLA":"Service Level Agreement",
            "RTO":"Recovery Time Objective",
            "RPO":"Recovery Point Objective",
            "BLAST_RADIUS":"Auswirkungen eines Ausfalls begrenzen"
        }
    },
    "portfolio": [
        {"id":"P-HEL","name":"Helios Datalake","phase":"Scale",
         "scope":"Unified ELT to Snowflake, dbt modeling, Kafka ingestion",
         "artifacts":[
            {"type":"runbook","title":"RB-ING-042 Ingestion Failover Runbook"},
            {"type":"rfc","title":"RFC-1287 Partitioning Strategy for Batch Loads"},
            {"type":"sla","title":"SLA-HEL-01 Datalake Availability 99.9%"}
         ]},
        {"id":"P-ORI","name":"Orion Edge Gateway","phase":"Build",
         "scope":"API gateway, rate limiting, auth integration",
         "artifacts":[
            {"type":"runbook","title":"RB-GW-011 Rolling Deployments (Blue/Green)"},
            {"type":"ticket","title":"GW-4821 MTLS Handshake Bug Analysis"},
            {"type":"sla","title":"SLA-ORI-02 p95 Latency < 120ms"}
         ]},
        {"id":"P-AEG","name":"Aegis IAM","phase":"Operate",
         "scope":"Enterprise SSO, RBAC, JIT access",
         "artifacts":[
            {"type":"runbook","title":"RB-IAM-075 Access Revocation Emergency"},
            {"type":"rfc","title":"RFC-903 Policy-as-Code Conventions"},
            {"type":"audit","title":"AUD-24-Q2 Access Review Summary"}
         ]},
        {"id":"P-ATL","name":"Atlas Mobile","phase":"Pilot",
         "scope":"Cross-platform app, feature flags, offline sync",
         "artifacts":[
            {"type":"design","title":"DS-ATLAS v2 Tokenized Components"},
            {"type":"runbook","title":"RB-MOB-021 Crash Loop Mitigation"}
         ]},
        {"id":"P-NIM","name":"Nimbus Observability","phase":"Build",
         "scope":"OpenTelemetry pipelines, SLOs, incident analytics",
         "artifacts":[
            {"type":"runbook","title":"RB-OBS-033 Alert Fatigue Tuning"},
            {"type":"rfc","title":"RFC-1114 Sampling Strategy for Traces"}
         ]},
        {"id":"P-QUA","name":"Quasar Billing","phase":"Scale",
         "scope":"Usage metering, cost allocation, anomaly detection",
         "artifacts":[
            {"type":"runbook","title":"RB-BIL-019 Revenue Recognition Edge Cases"},
            {"type":"ticket","title":"BIL-9324 Rounding Error in Tiered Pricing"}
         ]},
        {"id":"P-VES","name":"Vesta FinOps","phase":"Operate",
         "scope":"Cloud cost optimization & guardrails",
         "artifacts":[
            {"type":"rfc","title":"RFC-1502 Resource Quotas & Budgets"},
            {"type":"runbook","title":"RB-FIN-007 Idle Resource Reaper"}
         ]},
        {"id":"P-TIT","name":"Titan DR","phase":"Drill",
         "scope":"Disaster Recovery, multi-region failover",
         "artifacts":[
            {"type":"runbook","title":"RB-DR-001 Regional Failover Procedure"},
            {"type":"test","title":"TEST-DR-2025-Q1 GameDay Findings"}
         ]},
        {"id":"P-BOR","name":"Borealis ETL","phase":"Replatform",
         "scope":"Legacy ETL to modern ELT, CDC, schema evolution",
         "artifacts":[
            {"type":"rfc","title":"RFC-1711 CDC Strategy for Oracle → Snowflake"},
            {"type":"runbook","title":"RB-ETL-023 Backfill & Reconciliation"}
         ]},
        {"id":"P-PHX","name":"Phoenix Feature Store","phase":"Build",
         "scope":"Online/Offline feature serving, drift monitoring",
         "artifacts":[
            {"type":"rfc","title":"RFC-1419 Time-Travel Features"},
            {"type":"runbook","title":"RB-FS-034 Hotfix Rollback Procedure"}
         ]},
        {"id":"P-MER","name":"Mercury Messaging","phase":"Scale",
         "scope":"Event mesh, exactly-once semantics, backpressure control",
         "artifacts":[
            {"type":"rfc","title":"RFC-1222 Idempotency & Deduplication"},
            {"type":"runbook","title":"RB-MSG-044 Dead Letter Queue Draining"}
         ]},
        {"id":"P-POS","name":"Poseidon Networking","phase":"Operate",
         "scope":"Zero-trust networking, mTLS, service mesh policy",
         "artifacts":[
            {"type":"rfc","title":"RFC-1618 mTLS Policy Matrix"},
            {"type":"runbook","title":"RB-NET-029 Cert Rotation Checklist"}
         ]},
        {"id":"P-HER","name":"Hera QA Platform","phase":"Build",
         "scope":"Unified test orchestration, flaky test analytics",
         "artifacts":[
            {"type":"rfc","title":"RFC-1770 Risk-based Test Selection"},
            {"type":"runbook","title":"RB-QA-051 Release Candidate Gate"}
         ]},
        {"id":"P-CHR","name":"Chronos Scheduling","phase":"Operate",
         "scope":"Workload scheduling, cron lineage, failure budget",
         "artifacts":[
            {"type":"rfc","title":"RFC-1810 Retry Backoff Standards"},
            {"type":"runbook","title":"RB-SCH-013 Stuck Job Triage"}
         ]},
        {"id":"P-JAN","name":"Janus API Composition","phase":"Build",
         "scope":"BFF patterns, GraphQL/Federation, caching, authz",
         "artifacts":[
            {"type":"rfc","title":"RFC-1666 Edge Caching Guidelines"},
            {"type":"runbook","title":"RB-API-031 Token Expiry Incident SOP"}
         ]},
        {"id":"P-HYP","name":"Hyperion Cost Explorer","phase":"Pilot",
         "scope":"Real-time cost lenses, anomaly detectors, right-sizing",
         "artifacts":[
            {"type":"rfc","title":"RFC-1905 FinOps Lens Definitions"},
            {"type":"runbook","title":"RB-CST-027 Savings Plan Simulator"}
         ]}
    ]
}

PERSONAS = {
    "interviewers": [
        {"persona_id":"iv_struct_pm","role":"Senior Project Manager","interviewer_type":"structured","traits":["entscheidungsfreudig","prozessgetrieben","detailorientiert"]},
        {"persona_id":"iv_explore_data","role":"Lead Data Strategist","interviewer_type":"explorative","traits":["analytisch","systemisch","experimentierfreudig"]},
        {"persona_id":"iv_critical_sec","role":"Security Architect","interviewer_type":"critical","traits":["skeptisch","risikofokussiert","hartnäckig"]},
        {"persona_id":"iv_struct_ops","role":"Service Delivery Manager","interviewer_type":"structured","traits":["kundenfokus","SLA-strikt","pragmatisch"]},
        {"persona_id":"iv_explore_ux","role":"Staff UX Researcher","interviewer_type":"explorative","traits":["neugierig","empathisch","offen"]},
        {"persona_id":"iv_critical_qm","role":"Quality Manager","interviewer_type":"critical","traits":["perfektionistisch","beharrlich","präzise"]},
    ],
    "interviewees": [
        {"persona_id":"ee_cto","role":"CTO","traits":["visionär","knapp","priorisiert"]},
        {"persona_id":"ee_po","role":"Product Owner","traits":["kundenorientiert","priorisierung","tradeoffs"]},
        {"persona_id":"ee_pm","role":"Senior Project Manager","traits":["scope_control","risikoavers","stakeholder_mgmt"]},
        {"persona_id":"ee_sre","role":"Site Reliability Engineer","traits":["oncall","runbook-driven","pragmatisch"]},
        {"persona_id":"ee_devops","role":"DevOps Engineer","traits":["automation","IaC","pipelines"]},
        {"persona_id":"ee_da","role":"Data Engineer","traits":["ELT","dbt","airflow","lineage"]},
        {"persona_id":"ee_mlops","role":"MLOps Engineer","traits":["feature_store","model_ci_cd","drift"]},
        {"persona_id":"ee_sec","role":"Security Engineer","traits":["threat_modeling","vuln_mgmt","least_privilege"]},
        {"persona_id":"ee_qa","role":"QA Lead","traits":["test_strategy","risk_based_testing","traceability"]},
        {"persona_id":"ee_arch","role":"Cloud Architect","traits":["tradeoff","cost_perf","multi-region"]},
        {"persona_id":"ee_ux","role":"UX Lead","traits":["research","accessibility","design_system"]},
        {"persona_id":"ee_finops","role":"FinOps Analyst","traits":["savings","budgets","forecasting"]},
    ]
}

# ------------------ Prompts ------------------

BASE_SYSTEM = """
You are an interview orchestrator generating **long-form** synthetic IT interviews.
Rules:
- Fully fictional: no real companies or persons. Use the provided WORLD (company, projects) and PERSONAS.
- Make transcripts LONG and realistic; timestamps mm:ss strictly increasing; speakers: I, E, optional E2.
- Distribute key info across **START / MIDDLE / END** (anchors): early surface core facts, mid multi-hop links,
  late decisions/tradeoffs/risks with evidence (synthetic IDs, runbooks, tickets).
- Mix explicit knowledge (runbooks, SLAs, RFCs, tickets) and implicit knowledge (heuristics, unwritten rules).
- Keep language per 'language' parameter; for 'mix', aim for roughly **60% German / 40% English** at sentence level.
- Add light disfluencies if 'noise' is 'med' or 'high'. Avoid real brands or people.
"""

SCENARIO_AND_PLAN = """
Using WORLD and PERSONAS, synthesize a concrete scenario and a **long interview plan**.

Return JSON:
{
  "company": {"name":"...", "industry":"IT/Software"},
  "project": {"id":"...", "name":"...", "phase":"...", "scope":"..."},
  "participants": [
    {"id":"I","role":"Interviewer","persona_id":"..."},
    {"id":"E","role":"Interviewee","persona_id":"..."},
    {"id":"E2","role":"Interviewee","persona_id":"...", "optional":true}
  ],
  "plan": {
    "sections": [{"title":"...", "goal":"...", "sample_questions":["...","...","..."]}],
    "anchors": [
      {"id":"A-early","window":"start","content_hint":"surface core fact early"},
      {"id":"A-middle","window":"middle","content_hint":"non-trivial multi-hop link across subsystems"},
      {"id":"A-late","window":"end","content_hint":"decision/tradeoff/risk with evidence"}
    ],
    "style": {"interviewer_type":"${interviewer_type}", "noise":"${noise}", "code_switch":"${code_switch}"},
    "turns": {"min": ${min_turns}, "max": ${max_turns}},
    "duration_min": ${duration_min},
    "language": "${language}"
  }
}
Use project names from WORLD where reasonable; otherwise synthesize consistent names. For 'mix', enforce ~60/40 de/en distribution.
"""

ROLLOUT_CHUNK = """
Continue the interview and return a **JSON chunk**:
{
  "progress": {
    "next_turn_index": <int>,
    "next_ts_sec": <int>,
    "anchors_covered": ["A-early", ...],
    "memory": "<brief continuity summary>"
  },
  "transcript": [
    {"turn_index": <int>, "ts": "mm:ss", "speaker": "I|E|E2", "text": "..."}
  ]
}

Constraints:
- Generate **${turns_batch} NEW turns**, starting at turn_index=${start_index}, from timestamp=${start_ts_sec}s.
- Keep language='${language}' (for 'mix', keep ~60% German / 40% English by sentences).
- Ensure strictly increasing timestamps; target final ≈ ${duration_min} minutes.
- Honor remaining anchors (start/middle/late). Avoid repetition; add plausible technical detail (runbooks, tickets, SLAs).
- Do NOT repeat any meta object here.
"""

# ------------------ Helpers ------------------

def _mmss(sec: int) -> str:
    m = max(0, sec) // 60
    s = max(0, sec) % 60
    return f"{m:02d}:{s:02d}"

class TPMRateLimiter:
    """Simpler Token-per-Minute-Limiter (Best-Effort)."""
    def __init__(self, tpm_limit: int = 50000) -> None:
        self.tpm_limit = tpm_limit
        self.lock = threading.Lock()
        self.window_start = time.time()
        self.tokens_used = 0
    def consume(self, tokens: int) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.window_start
            if elapsed >= 60.0:
                self.window_start = now
                self.tokens_used = 0
            if self.tokens_used + tokens > self.tpm_limit:
                to_wait = 60.0 - elapsed + 0.05
                if to_wait > 0:
                    time.sleep(to_wait)
                self.window_start = time.time()
                self.tokens_used = 0
            self.tokens_used += tokens

class AzureClient:
    def __init__(self, max_output_tokens: Optional[int] = None):
        load_dotenv()
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if not endpoint or not api_key:
            raise RuntimeError("Fehlende .env Variablen: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.deployment = deployment
        if AzureOpenAI is None:
            raise RuntimeError("openai-Paket nicht installiert oder inkompatibel. Bitte `pip install openai>=1.40.0`")
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self.max_output_tokens = int(os.getenv("AZURE_MAX_OUTPUT_TOKENS", str(max_output_tokens or 32768)))
        self.rate = TPMRateLimiter(int(os.getenv("AZURE_TPM_LIMIT", "50000")))

    def chat_json_with_usage(self, messages: List[Dict[str, str]], temperature: float = 0.7, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str,int]]:
        kwargs = dict(model=self.deployment, messages=messages, temperature=temperature, max_tokens=self.max_output_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        # konservative Prompt-Token-Schätzung zur Rate-Limitierung
        rough_prompt_tokens = sum(len(m.get("content","").split()) for m in messages) // 0.75 if messages else 500
        self.rate.consume(int(rough_prompt_tokens))
        try:
            kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            usage_dict = dict(prompt_tokens=getattr(usage, "prompt_tokens", 0),
                              completion_tokens=getattr(usage, "completion_tokens", 0),
                              total_tokens=getattr(usage, "total_tokens", 0))
            add = max(0, usage_dict["total_tokens"] - int(rough_prompt_tokens))
            if add: self.rate.consume(add)
            return json.loads(content), usage_dict
        except Exception:
            resp = self.client.chat.completions.create(**{k:v for k,v in kwargs.items() if k != "response_format"})
            content = resp.choices[0].message.content
            start = content.find("{"); end = content.rfind("}")
            if start == -1 or end == -1:
                raise
            usage = getattr(resp, "usage", None)
            usage_dict = dict(prompt_tokens=getattr(usage, "prompt_tokens", 0),
                              completion_tokens=getattr(usage, "completion_tokens", 0),
                              total_tokens=getattr(usage, "total_tokens", 0))
            self.rate.consume(int(usage_dict["total_tokens"]))
            return json.loads(content[start:end+1]), usage_dict

# ------------------ Orchestrierung ------------------

def pick_personas(seed: Optional[int] = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    interviewer = rng.choice(PERSONAS["interviewers"])
    interviewee = rng.choice(PERSONAS["interviewees"])
    second = rng.choice([None] + PERSONAS["interviewees"]) if rng.random() < 0.25 else None
    if second and second["persona_id"] == interviewee["persona_id"]:
        second = None
    return {"interviewer": interviewer, "interviewee": interviewee, "interviewee2": second}

def build_plan(client: AzureClient, language: str, personas: Dict[str, Any],
               min_turns: int, max_turns: int, duration_min: int, noise: str, seed: Optional[int]) -> Dict[str, Any]:
    code_switch = "none" if language in ("de","en") else "strong"
    interviewer_type = personas["interviewer"].get("interviewer_type","structured")
    plan_prompt = Template(SCENARIO_AND_PLAN).substitute(
        interviewer_type=interviewer_type, noise=noise, code_switch=code_switch,
        min_turns=min_turns, max_turns=max_turns, duration_min=duration_min, language=language
    )
    messages = [
        {"role":"system","content": BASE_SYSTEM},
        {"role":"user","content": "WORLD:\n" + json.dumps(WORLD, ensure_ascii=False)},
        {"role":"user","content": "PERSONAS:\n" + json.dumps(personas, ensure_ascii=False)},
        {"role":"user","content": plan_prompt},
        {"role":"user","content": "Return ONLY JSON as specified."},
    ]
    payload, usage = client.chat_json_with_usage(messages, temperature=0.6, seed=seed)
    return payload

def rollout_step(client: AzureClient, plan_obj: Dict[str, Any], language: str, duration_min: int,
                 turns_batch: int, start_index: int, start_ts_sec: int, memory: str, seed: Optional[int]) -> Tuple[Dict[str, Any], Dict[str,int]]:
    user = Template(ROLLOUT_CHUNK).substitute(
        turns_batch=turns_batch, start_index=start_index, start_ts_sec=start_ts_sec,
        duration_min=duration_min, language=language
    )
    scenario = {"company": plan_obj.get("company"), "project": plan_obj.get("project")}
    messages = [
        {"role":"system","content": BASE_SYSTEM},
        {"role":"user","content": "Scenario:\n" + json.dumps(scenario, ensure_ascii=False)},
        {"role":"user","content": "Plan:\n" + json.dumps(plan_obj.get("plan", {}), ensure_ascii=False)},
        {"role":"user","content": "Continuity memory:\n" + (memory or "—")},
        {"role":"user","content": user},
    ]
    return client.chat_json_with_usage(messages, temperature=0.8, seed=seed)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --------- Sprach-Mix Scheduling ---------

def _parse_lang_mix(spec: str) -> Dict[str, float]:
    """
    spec format examples: "de=0.4,en=0.4,mix=0.2" or "de=40%,en=40%,mix=20%" or "de=4,en=4,mix=2"
    Returns normalized weights per language.
    """
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part: continue
        if "=" not in part: continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip().replace("%","")
        try:
            val = float(v)
        except ValueError:
            continue
        out[k] = val
    if not out:
        return {"de":0.4, "en":0.4, "mix":0.2}
    total = sum(out.values())
    if total <= 0:
        return {"de":0.4, "en":0.4, "mix":0.2}
    return {k: (v/total) for k,v in out.items() if k in ("de","en","mix")}

def build_language_schedule(count: int, default_lang: str, mix_spec: Optional[str], seed: Optional[int]) -> List[str]:
    """
    Deterministische Verteilung über count Items.
    Hamilton-Apportionment + Round-Robin, damit die Sprachen über den Index verteilt sind.
    """
    if not mix_spec:
        return [default_lang]*count
    weights = _parse_lang_mix(mix_spec)
    langs = ["de","en","mix"]
    # Zielanzahl je Sprache (Hamilton)
    quotas = {l: weights.get(l,0.0) * count for l in langs}
    base = {l: int(quotas[l] // 1) for l in langs}
    remainder = {l: quotas[l] - base[l] for l in langs}
    allocated = sum(base.values())
    remaining = max(0, count - allocated)
    # verteile Rest nach größtem Rest
    for l,_ in sorted(remainder.items(), key=lambda kv: kv[1], reverse=True):
        if remaining <= 0: break
        base[l] += 1
        remaining -= 1
    # Round-Robin-Sequenz
    seq = []
    left = {l: base[l] for l in langs}
    while sum(left.values()) > 0:
        for l in langs:
            if left[l] > 0:
                seq.append(l); left[l] -= 1
    # deterministische leichte Durchmischung mit Seed
    rng = random.Random(seed if seed is not None else 0xC0FFEE)
    # einfache Streuung: swap in festen Intervallen
    for i in range(0, len(seq), 7):
        if i+3 < len(seq) and rng.random() < 0.5:
            seq[i], seq[i+3] = seq[i+3], seq[i]
    return seq[:count]

# ------------------ Haupt-Generator ------------------

# ------------------ Haupt-Generator ------------------

def generate_one(
    out_dir: str, language: str = "de",
    min_turns: int = 100, max_turns: int = 360,
    duration_min: int = 90, noise: str = "med",
    turns_per_step: int = 16, initial_steps: int = 20,
    target_output_tokens: int = 40000, max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    os.makedirs(os.path.join(out_dir, "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

    client = AzureClient(max_output_tokens=max_output_tokens)
    personas = pick_personas(seed=seed)
    logging.info(f"[Setup] Personas selected — interviewer={personas['interviewer']['persona_id']} | interviewee={personas['interviewee']['persona_id']} | lang={language}")
    plan_obj = build_plan(client, language, personas, min_turns, max_turns, duration_min, noise, seed)

    uid = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    meta_path = os.path.join(out_dir, "meta", f"meta_{uid}.json")
    tr_path = os.path.join(out_dir, "transcripts", f"transcript_{uid}.jsonl")

    meta_obj = {
        "id": uid,
        "language": language,
        "params": {
            "min_turns": min_turns, "max_turns": max_turns, "duration_min": duration_min,
            "noise": noise, "mode": "long-rollout", "seed": seed,
            "turns_per_step": turns_per_step,
            "initial_steps": initial_steps,
            "target_output_tokens": target_output_tokens, "max_output_tokens": max_output_tokens
        },
        "personas": personas,
        "world_name": WORLD["company"]["name"],
        "model": {"provider": "azure-openai", "deployment_env": os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-4o")},
        "plan": plan_obj
    }
    logging.info(f"[Start] Transcript {uid} — lang={language}, target_tokens≈{target_output_tokens}, max_turns={max_turns}, duration≈{duration_min}m")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)

    start_index = 1
    start_ts_sec = 0
    memory = ""
    last_ts = 0
    all_turns = 0
    total_completion_tokens = 0
    anchors_covered = set()

    step = 0
    # Loop bis Ziel-Completion erreicht ODER max_turns droht
    while True:
        if step >= initial_steps and total_completion_tokens >= target_output_tokens:
            break
        if all_turns >= max_turns:
            break
        payload, usage = rollout_step(client, plan_obj, language, duration_min, turns_per_step,
                                      start_index, start_ts_sec, memory, seed=None if seed is None else seed + step + 1)

        chunk = payload.get("transcript", []) if isinstance(payload, dict) else []
        rows = []
        for item in chunk:
            ts = item.get("ts")
            try:
                if not ts or len(ts.split(":")) != 2:
                    last_ts += 15
                    ts = _mmss(last_ts)
                else:
                    mm, ss = ts.split(":"); last_ts = int(mm) * 60 + int(ss)
            except Exception:
                last_ts += 15
                ts = _mmss(last_ts)
            rows.append({"ts": ts, "speaker": item.get("speaker", "I"), "text": item.get("text", "")})
            all_turns += 1
            if all_turns >= max_turns:
                break
        write_jsonl(tr_path, rows)

        prog = payload.get("progress", {}) if isinstance(payload, dict) else {}
        start_index = int(prog.get("next_turn_index", start_index + turns_per_step))
        start_ts_sec = int(prog.get("next_ts_sec", last_ts + 15))
        memory = prog.get("memory", memory)
        for a in (prog.get("anchors_covered", []) or []):
            anchors_covered.add(a)

        comp = int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
        total_completion_tokens += comp
        pct = min(100, int(100 * (total_completion_tokens / max(1, target_output_tokens))))
        try:
            next_ts = _mmss(start_ts_sec)
        except Exception:
            next_ts = str(start_ts_sec)
        logging.info(f"[{uid}] step={step+1} +{comp} tok | total={total_completion_tokens}/{target_output_tokens} ({pct}%) | turns={all_turns}/{max_turns} | next_turn={start_index} @ {next_ts}")
        step += 1

    meta_obj["rollout"] = {
        "total_completion_tokens": total_completion_tokens,
        "all_turns": all_turns,
        "anchors_covered": sorted(list(anchors_covered)),
        "steps_executed": step
    }
    logging.info(f"[Done] Transcript {uid} — tokens={total_completion_tokens}, turns={all_turns}, steps={step}, anchors={len(anchors_covered)}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)

    return {"id": uid, "meta": meta_path, "transcript": tr_path}

# ------------------ CLI ------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Lange, informationsdichte Interview-Transkripte generieren (Azure OpenAI).")
    p.add_argument("--count", type=int, default=1, help="Anzahl der Transkripte")
    p.add_argument("--out", type=str, default="data", help="Output-Verzeichnis")
    p.add_argument("--parallel", type=int, default=1, help="Anzahl paralleler Threads für die Generierung")
    # Sprachsteuerung
    p.add_argument("--lang", type=str, default="de", choices=["de","en","mix"], help="Sprache für alle Transkripte (falls --lang-mix nicht gesetzt)")
    p.add_argument("--lang-mix", type=str, default="de=0.4,en=0.4,mix=0.2",
                   help="Empfohlenes Verhältnis über den Batch, z. B. 'de=0.4,en=0.4,mix=0.2' oder 'de=40%,en=40%,mix=20%'")
    # Längensteuerung
    p.add_argument("--min-turns", type=int, default=260)
    p.add_argument("--max-turns", type=int, default=360)
    p.add_argument("--duration-min", type=int, default=90)
    p.add_argument("--noise", type=str, default="med", choices=["low","med","high"])
    p.add_argument("--initial-steps", type=int, default=20)
    p.add_argument("--turns-per-step", type=int, default=16)
    p.add_argument("--target-output-tokens", type=int, default=40000)
    p.add_argument("--max-output-tokens", type=int, default=32768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL","INFO"), choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging-Level")

    ns = p.parse_args(argv)
    # Logging konfigurieren
    logging.basicConfig(level=getattr(logging, str(ns.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # Sprachverteilung bauen
    schedule = build_language_schedule(ns.count, ns.lang, ns.lang_mix, ns.seed)
    logging.info(f"[Batch] Starte Generierung von {ns.count} Transkript(en). Schedule: {', '.join(s.upper() for s in schedule)}")
    base_seed = ns.seed if ns.seed is not None else random.randint(1, 10_000_000)

    results = []
    parallelism = max(1, int(getattr(ns, "parallel", 1) or 1))

    def _worker(idx_lang: int, lang: str):
        remaining_after = len(schedule) - (idx_lang + 1)
        logging.info(f"[Batch] Starte {idx_lang+1}/{len(schedule)} — Sprache={lang} — verbleibend nach diesem: {remaining_after}")
        res = generate_one(
            out_dir=ns.out, language=lang,
            min_turns=ns.min_turns, max_turns=ns.max_turns, duration_min=ns.duration_min, noise=ns.noise,
            turns_per_step=ns.turns_per_step, initial_steps=ns.initial_steps,
            target_output_tokens=ns.target_output_tokens, max_output_tokens=ns.max_output_tokens,
            seed=base_seed + idx_lang
        )
        print(f"[OK] {lang.upper()}  {res['id']} -> {res['transcript']}")
        logging.info(f"[Batch] Fertig {idx_lang+1}/{len(schedule)} — {res['id']} ({lang})")
        return idx_lang, res

    if parallelism == 1 or len(schedule) == 1:
        for i, lang in enumerate(schedule):
            _, res = _worker(i, lang)
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="transcript") as ex:
            futures = {ex.submit(_worker, i, lang): i for i, lang in enumerate(schedule)}
            for fut in as_completed(futures):
                idx, res = fut.result()
                results.append(res)

    print(f"Fertig. {len(results)} Transkript(e). Sprachen (Plan): {ns.lang_mix}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
