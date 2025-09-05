import os, hashlib, json
from typing import List, Dict, Any
import numpy as np
from .config import Config
from .log import warn, info

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

class AzureClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = None
        if not cfg.dry_run:
            if AzureOpenAI is None:
                warn("openai Paket nicht verfügbar – schalte in DRY_RUN.")
                self.cfg.dry_run = True
            elif not (cfg.endpoint and cfg.api_key):
                warn("Azure OpenAI ENV unvollständig – schalte in DRY_RUN.")
                self.cfg.dry_run = True
            else:
                self.client = AzureOpenAI(
                    api_key=cfg.api_key,
                    api_version=cfg.api_version,
                    azure_endpoint=cfg.endpoint,
                )

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.cfg.dry_run:
            # Deterministischer Fallback auf Basis von Hashes
            vecs = []
            for t in texts:
                h = hashlib.sha256(t.encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(h[:8], "big", signed=False))
                vecs.append(rng.standard_normal(1536).astype("float32"))
            return np.vstack(vecs)
        else:
            resp = self.client.embeddings.create(
                model=self.cfg.embed_deployment,
                input=texts
            )
            vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
            return np.vstack(vecs)

    def judge(self, question: str, context: str) -> Dict[str, Any]:
        """Return a dict with fields: score (0..1), verdict ('YES'/'NO'), rationale (str)."""
        if self.cfg.dry_run:
            # Heuristik: Cosine über Hash-basierte Vektoren
            import numpy as np, hashlib
            def hv(s):
                h = hashlib.sha256(s.encode('utf-8')).digest()
                rng = np.random.default_rng(int.from_bytes(h[:8], 'big', signed=False))
                return rng.standard_normal(256).astype('float32')
            qv, cv = hv(question), hv(context)
            sim = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv) + 1e-9))
            score = max(0.0, min(1.0, 0.5 + 0.5*sim))
            return {"score": score, "verdict": "YES" if score >= 0.6 else "NO", "rationale": "heuristic"}
        else:
            system = (
                "You are a strict evaluator. Decide if the given CONTEXT contains enough information to answer the QUESTION. "
                'Reply with JSON: {"score": 0..1, "verdict": "YES|NO", "rationale": "..."}. '
                "Be conservative; require concrete evidence in the context."
            )
            user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
            resp = self.client.chat.completions.create(
                model=self.cfg.chat_deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            try:
                obj = json.loads(text)
            except Exception:
                obj = {"score": 0.0, "verdict": "NO", "rationale": text[:4000]}
            return obj
