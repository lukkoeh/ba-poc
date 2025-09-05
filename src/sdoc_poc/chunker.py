from typing import List

def chunk_text(text: str, max_words: int = 220, overlap_words: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(len(words), i + max_words)
        chunk = ' '.join(words[i:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(words):
            break
        i = end - overlap_words
        if i < 0: i = 0
    return chunks
