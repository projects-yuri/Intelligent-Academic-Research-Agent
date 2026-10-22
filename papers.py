"""Biblioteca local de dez artigos verificados no arXiv."""

from functools import lru_cache
from pathlib import Path
import json
import re
import unicodedata

DATA_FILE = Path(__file__).resolve().parent / "data" / "papers.json"

STOPWORDS = {
    "de", "da", "do", "dos", "das", "e", "em", "para", "com", "que", "um", "uma", "o", "a",
    "os", "as", "sobre", "quais", "qual", "por", "mais", "the", "and", "for", "of", "on", "in",
    "what", "which", "are", "news", "fake", "artigo", "artigos", "estudo", "estudos",
}

TRANSLATIONS = {
    "desinformacao": ["misinformation", "disinformation"],
    "brasileiro": ["brazilian", "portuguese"],
    "brasil": ["brazilian", "portuguese"],
    "portugues": ["portuguese", "brazilian"],
    "imagens": ["image", "multimodal"],
    "imagem": ["image", "multimodal"],
    "dados": ["dataset", "benchmark"],
    "base": ["dataset"],
    "generalizacao": ["generalization"],
    "revisao": ["survey"],
    "comparar": ["comparison"],
}


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


@lru_cache(maxsize=1)
def get_library() -> list[dict]:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_local_paper(reference: str) -> dict | None:
    cleaned = str(reference).strip().lower().rstrip("/")
    cleaned = cleaned.replace("https://arxiv.org/abs/", "").replace("arxiv:", "")
    cleaned = cleaned.replace("https://arxiv.org/pdf/", "").removesuffix(".pdf")
    for item in get_library():
        if cleaned == item["id"].lower():
            return item.copy()
    return None


def search_library(query: str, limit: int = 5) -> list[dict]:
    query_words = set(_normalize(query).split()) - STOPWORDS
    for word in list(query_words):
        query_words.update(TRANSLATIONS.get(word, []))
    if not query_words:
        return get_library()[:limit]
    ranked = []
    for article in get_library():
        title = _normalize(article["title"])
        content = _normalize(" ".join([article["topic"], article["summary_pt"],
                                       *article["methods"], *article["keywords"], *article["datasets"]]))
        title_tokens = set(title.split())
        content_tokens = set(content.split())
        score = sum(3 for w in query_words if w in title_tokens) + sum(1 for w in query_words if w in content_tokens)
        if score:
            ranked.append((score, article["year"], article))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [article for _, _, article in ranked[:limit]]
