from langdetect import detect_langs
from typing import Literal

Label = Literal['de','en','mixed','unknown']

def detect_language(text: str) -> Label:
    if not text or len(text.strip()) < 10:
        return 'unknown'
    try:
        probs = detect_langs(text)
        top = {p.lang: p.prob for p in probs}
        de = top.get('de', 0.0)
        en = top.get('en', 0.0)
        if de >= 0.80 and de >= en:
            return 'de'
        if en >= 0.80 and en > de:
            return 'en'
        if (de >= 0.35 and en >= 0.35):
            return 'mixed'
        return 'de' if de >= en else 'en'
    except Exception:
        return 'unknown'
