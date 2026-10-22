"""As três ferramentas reais do agente ReAct acadêmico.

1. search_papers: consulta a coleção local e, opcionalmente, o OpenAlex.
2. extract_information: abre metadados/resumo do artigo por identificador.
3. compare_studies: estrutura uma comparação fundamentada entre 2 a 5 artigos.

O conteúdo recuperado é dado externo, nunca instrução para o agente.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any

import requests
from langchain_core.tools import tool
from papers import get_local_paper, search_library

TIMEOUT = 8
USER_AGENT = "MossScholar/1.0 (academic demo; public metadata only)"
OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "https://export.arxiv.org/api/query"


def _http_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(
        url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    return response.json()


def _abstract_from_index(index: dict | None) -> str:
    """Reconstrói resumo textual a partir do índice invertido do OpenAlex."""
    if not index:
        return ""
    words: dict[int, str] = {}
    for term, positions in index.items():
        for position in positions:
            words[int(position)] = term
    return " ".join(words.get(i, "") for i in range(max(words, default=-1) + 1)).strip()


def _normalize_openalex(work: dict) -> dict:
    identifier = str(work.get("id") or "").split("/")[-1]
    authors = [entry["author"]["display_name"] for entry in work.get("authorships", [])
               if entry.get("author", {}).get("display_name")]
    location = work.get("best_oa_location") or {}
    abstract = _abstract_from_index(work.get("abstract_inverted_index"))
    return {
        "id": identifier,
        "title": work.get("title") or work.get("display_name") or "Sem título",
        "authors": authors[:12],
        "year": work.get("publication_year"),
        "url": work.get("doi") or work.get("id") or "",
        "pdf_url": location.get("pdf_url") or "",
        "topic": "Resultado externo (OpenAlex)",
        "methods": [],
        "datasets": [],
        "summary_pt": "",
        "abstract": abstract[:5500],
        "source": "OpenAlex",
    }


def _openalex_params(extra: dict) -> dict:
    params = dict(extra)
    key = os.getenv("OPENALEX_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    return params


def _search_openalex(query: str, limit: int) -> list[dict]:
    data = _http_json(OPENALEX_URL, _openalex_params({
        "search": query, "per_page": limit, "filter": "is_oa:true"
    }))
    return [_normalize_openalex(item) for item in data.get("results", [])]


def _get_openalex(identifier: str) -> dict:
    if not re.fullmatch(r"W\d{5,20}", identifier, re.IGNORECASE):
        raise ValueError("ID do OpenAlex inválido. Use, por exemplo, W2741809807.")
    result = _http_json(f"{OPENALEX_URL}/{identifier.upper()}", _openalex_params({}))
    return _normalize_openalex(result)


@lru_cache(maxsize=100)
def _get_arxiv_abstract(article_id: str) -> str:
    if not re.fullmatch(r"\d{4}\.\d{4,5}", article_id):
        return ""
    response = requests.get(
        ARXIV_URL, params={"id_list": article_id}, timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is None:
        return ""
    node = entry.find("{http://www.w3.org/2005/Atom}summary")
    return " ".join(node.text.split())[:5500] if node is not None and node.text else ""


def _article(reference: str, *, fetch_abstract: bool = False) -> dict:
    local = get_local_paper(reference)
    if local:
        local["source"] = "Biblioteca curada · arXiv"
        local["abstract"] = ""
        local["abstract_status"] = "Não consultado; a síntese local não é o resumo original."
        if fetch_abstract:
            try:
                local["abstract"] = _get_arxiv_abstract(local["id"])
                local["abstract_status"] = "Resumo original do arXiv" if local["abstract"] else "Resumo indisponível; veja a síntese local."
            except (requests.RequestException, ET.ParseError):
                local["abstract_status"] = "Falha na consulta ao arXiv; disponível apenas a síntese local."
        return local

    code = str(reference).strip().split("/")[-1].upper()
    if re.fullmatch(r"W\d{5,20}", code):
        return _get_openalex(code)
    raise ValueError("Artigo não encontrado. Use um ID arXiv da biblioteca ou um ID OpenAlex (W...).")


def _lightweight(data: dict) -> dict:
    return {key: data.get(key) for key in (
        "id", "title", "authors", "year", "url", "pdf_url", "topic", "methods", "datasets", "summary_pt", "source"
    )}


@tool
def search_papers(query: str, max_results: int = 5) -> str:
    """Busca artigos científicos por assunto, autor ou método. Primeiro pesquisa os dez artigos locais; com OPENALEX_API_KEY, consulta também o OpenAlex. Retorna IDs para uso nas outras ferramentas."""
    if not query.strip():
        return json.dumps({"error": "Informe um tema para a pesquisa."}, ensure_ascii=False)
    limit = min(max(int(max_results), 1), 8)
    local = [{**_lightweight(p), "source": "Biblioteca curada · arXiv"} for p in search_library(query, limit)]
    found = list(local)
    warnings = []
    if os.getenv("OPENALEX_API_KEY", "").strip():
        try:
            remote = _search_openalex(query, limit)
            existing = {p["title"].lower().strip() for p in found}
            for paper in remote:
                if paper["title"].lower().strip() not in existing:
                    found.append({**_lightweight(paper), "abstract_excerpt": paper["abstract"][:650]})
                    existing.add(paper["title"].lower().strip())
        except (requests.RequestException, ValueError, KeyError) as exc:
            warnings.append(f"Não foi possível consultar OpenAlex ({type(exc).__name__}); resultados locais preservados.")
    else:
        warnings.append("Busca limitada à biblioteca inicial de dez artigos; configure OPENALEX_API_KEY para ampliar.")
    if not found:
        warnings.append("Nenhum artigo da biblioteca local correspondeu ao termo pesquisado.")
    return json.dumps({"query": query, "total": len(found), "papers": found[:limit + (limit if os.getenv('OPENALEX_API_KEY') else 0)], "notices": warnings}, ensure_ascii=False)


@tool
def extract_information(reference: str) -> str:
    """Obtém informações de um artigo por ID: título, autores, ano, link, métodos, conjuntos de dados e resumo original quando disponível. Aceita IDs arXiv locais (ex.: 2309.11052) ou OpenAlex (W...)."""
    try:
        article = _article(reference, fetch_abstract=True)
        return json.dumps(article, ensure_ascii=False)
    except (ValueError, requests.RequestException, KeyError) as exc:
        return json.dumps({"error": str(exc), "reference": reference}, ensure_ascii=False)


@tool
def compare_studies(references: list[str]) -> str:
    """Compara de dois a cinco artigos pelos IDs, organizando temas, técnicas, conjuntos de dados e pontos comuns. Requer referências existentes; não inventa resultados experimentais."""
    ids = list(dict.fromkeys(ref.strip() for ref in references if ref.strip()))
    if not 2 <= len(ids) <= 5:
        return json.dumps({"error": "Forneça de 2 a 5 IDs distintos."}, ensure_ascii=False)
    studies: list[dict[str, Any]] = []
    errors = []
    for ref in ids:
        try:
            studies.append(_article(ref, fetch_abstract=False))
        except (ValueError, requests.RequestException, KeyError) as exc:
            errors.append({"reference": ref, "error": str(exc)})
    shared = set(studies[0].get("methods") or []) if studies else set()
    for study in studies[1:]:
        shared.intersection_update(study.get("methods") or [])
    return json.dumps({
        "studies": [{
            "id": s["id"], "title": s["title"], "year": s["year"], "url": s["url"],
            "focus": s.get("topic"), "methods": s.get("methods", []),
            "datasets": s.get("datasets", []), "summary": s.get("summary_pt", ""),
            "abstract_excerpt": s.get("abstract", "")[:2200], "source": s.get("source")
        } for s in studies],
        "shared_methods_in_metadata": sorted(shared),
        "notices": [
            "Uma lista vazia de métodos significa metadados insuficientes, não ausência de método.",
            "Os textos locais são sínteses curadas; para conclusões detalhadas consulte o artigo completo.",
            "O agente deve explicitar quando as fontes não permitem uma comparação conclusiva."
        ],
        "errors": errors,
    }, ensure_ascii=False)


TOOLS = [search_papers, extract_information, compare_studies]
