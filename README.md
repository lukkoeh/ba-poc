# SDOC – Semantic Documents Pipeline (PoC)

Proof-of-Concept für die Bachelorarbeit: End-to-end Pipeline **inkl. Evaluation**, ausführbar mit **einem einzigen Befehl**. 
Vollständig **zwischengespeichert** und **fortsetzbar**. Azure OpenAI wird via **.env** angebunden.

## Schnellstart
1. Python 3.10+.
2. `cp .env.example .env` und die Azure-Variablen setzen.
3. Abhängigkeiten installieren:  
   ```bash
   pip install -r requirements.txt
   ```
4. Deine 250 Dateien ablegen:
   - Transkripte nach `data/transcripts/`
   - Metadateien nach `data/meta/`
   > Konvention: Dateien wie `transcript_YYYYMMDD_HHMMSS_<ID>.jsonl` und `meta_YYYYMMDD_HHMMSS_<ID>.json`.
5. **Alles starten (Pipeline + Evaluation + Subagent):**  
   ```bash
   ./run.sh
   ```
   oder
   ```bash
   make run
   ```

## Was passiert?
- **Ingest**: Zuordnung Transcript ↔ Meta über `<ID>` aus Dateinamen. Robust gegenüber `.jsonl`, `.json` oder `.txt` Transkripten.
- **Detect**: Sprachdetektion (DE/EN/Mixed) auf Dokumentebene.
- **Chunk**: Semantische Segmente (wortbasiert, mit Overlap).
- **Embed**: Azure OpenAI Embeddings (Cache, deterministisches Fallback via `DRY_RUN=1`).
- **Index**: Einfacher Vektorindex (brute-force, NumPy). Keine nativen Abhängigkeiten nötig.
- **Retrieve**: Dokument-Scoped & Global Retrieval.
- **Evaluate**:
    - **Retrieval@K gegen Meta-Fragen** (aus `plan.sections[*].sample_questions`)
    - **LLM-as-Judge** (Ja/Nein/Score, Cache); Fallback ohne API → heuristische Scores.
    - **Anchor-Check** (`A-early`, `A-middle`, `A-late`) via LLM-Tagging vs. `anchors_covered`.
    - **Sprachverteilung** vs. Erwartung (40% DE / 40% EN / 20% Mixed).
- **Subagent**: LLM-gestützter Code-Check (statisch + Review-Prompt). Ergebnisse in `artifacts/subagent_report.md`.

## Resumierbarkeit & Caching
- **SQLite** in `artifacts/state.db` (Docs, Chunks, Embeddings, Urteile).
- **Vektoren** in `artifacts/vecs/*.npy` (pro Modell & Chunk-Hash).
- Wiederholte Läufe **skipping** bereits verarbeitete Einheiten. Änderungen an Dateien werden via Hash erkannt.

## Konfiguration
- `.env` für Azure-Keys und Deployments.  
- `DRY_RUN=1` ermöglicht lokale Smoke-Tests ohne API.
- `MAX_DOCS` begrenzt die Verarbeitung (0 = unbegrenzt).

## Ausgaben
- **Berichte**: `artifacts/report.json` + `artifacts/report.md`
- **Index**: `artifacts/index.jsonl` (Metadaten) + `artifacts/vecs/*.npy`
- **Logs**: reichlich Konsole + `artifacts/run.log`

## Ordner
```text
sdoc_poc/
├─ data/
│  ├─ transcripts/   # <- hierhin kommen deine 250 Transkripte
│  └─ meta/          # <- hierhin kommen die Metadateien
├─ samples/          # Beispiel-Dateien (falls vorhanden)
├─ src/sdoc_poc/     # Pipeline-Code
├─ artifacts/        # Zwischenergebnisse, Caches, Reports
├─ run.py            # Single-Entry CLI
└─ run.sh
```
