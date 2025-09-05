# Simple Transcript Generator (single script)

Ein **einfaches Python3-Skript** erzeugt **lange, informationsdichte** Interview-Transkripte mit Azure OpenAI (gpt-4o / gpt-5-chat).
- Personen, Company und Projekte sind **eingebettet** (fiktiv: *Novereon Systems GmbH* mit 8 Projekten).
- Mehrstufiger Rollout (Schritte × Turns) verteilt **Anker** über Anfang / Mitte / Ende.
- Output: `transcripts/*.jsonl` (eine Zeile pro Turn) + `meta/*.json` (Plan, Personas, Params).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade openai python-dotenv
cp .env.example .env
# .env ausfüllen
```

`.env`:
```
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_DEPLOYMENT=gpt-5-chat   # oder gpt-4o
AZURE_TPM_LIMIT=50000
AZURE_MAX_OUTPUT_TOKENS=16384        # ggf. auf Modellmaximum setzen
```

## Nutzung

```bash
python gen_transcripts.py --count 1 --out data --lang de   --min-turns 100 --max-turns 160 --duration-min 60   --steps 12 --turns-per-step 15 --target-output-tokens 24000   --max-output-tokens 16384
```

Für 250 Stück:
```bash
python gen_transcripts.py --count 250 --out data_250 --lang de   --min-turns 100 --max-turns 160 --duration-min 60   --steps 12 --turns-per-step 15 --target-output-tokens 24000   --max-output-tokens 16384
```

## Hinweise
- Die Prompts streuen Inhalte effizient über den Verlauf (Anker), adressieren **Position Bias** und forcieren **Multi-Hop** in der Mitte.
- Timestamps sind streng monoton steigend (`mm:ss`).
- Du kannst Personas/Company inline im Skript verändern (`WORLD`, `PERSONAS`).

