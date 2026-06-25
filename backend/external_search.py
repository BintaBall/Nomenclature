"""
external_search.py — Moteur de recherche externe
==================================================
Déclenché quand la similarité interne retourne des résultats faibles.
Sources : GitHub API · HuggingFace Hub · ArXiv Atom

Pourquoi PAS DuckDuckGo :
  L'API Instant Answer de DDG retourne uniquement des résumés Wikipedia.
  Elle ne connaît pas les repos GitHub ou HuggingFace.
  Pour des projets IA, GitHub + HF + ArXiv couvrent 99% du corpus utile.

Dépendances : requests feedparser (déjà dans requirements.txt)
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("external_search")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

_GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}
_HF_HEADERS = {
    "Accept": "application/json",
    **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
}

SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() != "false"

# ─────────────────────────────────────────────────────────────────────────────
# Seuil de déclenchement de la recherche externe
# Si le meilleur score de similarité interne est sous ce seuil → on déclenche
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TRIGGER_SCORE = 0.45   # cosine TF-IDF — ajuster selon tes résultats

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExternalResult:
    title:        str
    description:  str
    url:          str
    source:       str          # "github" | "huggingface" | "arxiv"
    author:       str  = ""
    stars:        int  = 0
    has_paper:    bool = False
    paper_url:    str  = ""
    domain:       str  = ""
    tags:         list = field(default_factory=list)
    published_at: str  = ""
    relevance:    float = 0.0   # score maison [0-1]


@dataclass
class ExternalSearchResult:
    query:           str
    results:         list[ExternalResult]
    sources_used:    list[str]
    total_found:     int
    triggered_by:    str   # "low_similarity" | "manual" | "new_domain"
    best_relevance:  float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_get(url: str, headers: dict, params: dict = None,
              timeout: int = 12) -> Optional[requests.Response]:
    NO_RETRY = {400, 401, 403, 404, 410, 412, 422, 451}
    try:
        r = requests.get(url, headers=headers, params=params or {},
                         timeout=timeout, verify=SSL_VERIFY)
        if r.status_code in NO_RETRY:
            log.debug(f"HTTP {r.status_code} (no retry): {url}")
            return None
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 30))
            log.warning(f"Rate limit — attente {wait}s")
            time.sleep(wait)
            return None
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            log.debug(f"Réponse HTML reçue (proxy?) — {url}")
            return None
        return r
    except requests.exceptions.SSLError:
        log.debug(f"SSL error — ajoute SSL_VERIFY=false dans .env")
        return None
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        log.debug(f"Connexion échouée: {e}")
        return None
    except requests.RequestException as e:
        log.warning(f"Requête échouée: {e}")
        return None


def _relevance_score(query: str, text: str) -> float:
    """
    Score de pertinence maison basé sur le chevauchement de mots-clés.
    Simple mais efficace pour re-ranker des résultats externes.
    """
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    text_words  = set(re.findall(r'\b\w{3,}\b', text.lower()))
    if not query_words:
        return 0.0
    overlap = len(query_words & text_words)
    return round(min(overlap / len(query_words), 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# GitHub Search
# ─────────────────────────────────────────────────────────────────────────────

class GitHubSearcher:
    BASE = "https://api.github.com"

    def search(self, query: str, max_results: int = 10) -> list[ExternalResult]:
        """
        Recherche des repos GitHub pertinents pour la requête.
        Stratégie : recherche dans description + README, triée par stars.
        """
        results = []

        # Construction de la query GitHub — on ajoute le contexte ML
        gh_query = f"{query} language:Python"
        r = _safe_get(
            f"{self.BASE}/search/repositories",
            _GITHUB_HEADERS,
            {
                "q": gh_query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(max_results * 2, 30),
            },
        )
        if r is None:
            log.warning("GitHub — inaccessible ou rate limit")
            return []

        try:
            items = r.json().get("items", [])
        except Exception:
            return []

        for item in items:
            desc = (item.get("description") or "").strip()
            if not desc:
                continue

            combined = f"{item.get('name','')} {desc} {' '.join(item.get('topics',[]))}"
            rel = _relevance_score(query, combined)

            results.append(ExternalResult(
                title       = item.get("full_name", ""),
                description = desc[:300],
                url         = item.get("html_url", ""),
                source      = "github",
                author      = (item.get("owner") or {}).get("login", ""),
                stars       = item.get("stargazers_count", 0),
                has_paper   = bool(item.get("homepage")),
                tags        = item.get("topics", [])[:8],
                published_at= item.get("created_at", "")[:10],
                relevance   = rel,
            ))

        # Re-rank par pertinence
        results.sort(key=lambda x: (-x.relevance, -x.stars))
        return results[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Search
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceSearcher:
    BASE = "https://huggingface.co/api"

    def search_models(self, query: str, max_results: int = 8) -> list[ExternalResult]:
        """
        Recherche des modèles HuggingFace via l'API Hub.
        Utilise le paramètre `search` supporté nativement.
        """
        r = _safe_get(
            f"{self.BASE}/models",
            _HF_HEADERS,
            {
                "search":    query,
                "sort":      "downloads",
                "direction": -1,
                "limit":     min(max_results * 2, 30),
                "cardData":  True,
                "full":      True,
            },
        )
        if r is None:
            log.warning("HuggingFace — inaccessible ou rate limit")
            return []

        try:
            items = r.json()
            if not isinstance(items, list):
                return []
        except Exception:
            return []

        results = []
        for item in items:
            model_id = item.get("modelId") or item.get("id", "")
            card     = item.get("cardData") or {}
            desc     = (card.get("description") or item.get("description") or "").strip()

            # Si pas de description, synthétise depuis les tags
            if not desc:
                pipeline = item.get("pipeline_tag") or ""
                tags     = item.get("tags", [])[:5]
                desc     = f"{pipeline} model — {', '.join(tags)}" if pipeline else ", ".join(tags)

            if not desc or len(desc) < 10:
                continue

            tags_raw = item.get("tags", [])[:10]
            combined = f"{model_id} {desc} {' '.join(tags_raw)}"
            rel      = _relevance_score(query, combined)

            # Détection ArXiv dans les tags
            arxiv_url = ""
            for tag in tags_raw:
                if tag.lower().startswith("arxiv:"):
                    arxiv_id  = tag[6:].strip()
                    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                    break

            results.append(ExternalResult(
                title       = model_id.split("/")[-1] if "/" in model_id else model_id,
                description = desc[:300],
                url         = f"https://huggingface.co/{model_id}",
                source      = "huggingface",
                author      = item.get("author", ""),
                stars       = item.get("likes", 0) or 0,
                has_paper   = bool(arxiv_url),
                paper_url   = arxiv_url,
                domain      = item.get("pipeline_tag", ""),
                tags        = tags_raw,
                published_at= (item.get("createdAt") or "")[:10],
                relevance   = rel,
            ))

        results.sort(key=lambda x: (-x.relevance, -x.stars))
        return results[:max_results]

    def search_spaces(self, query: str, max_results: int = 5) -> list[ExternalResult]:
        """
        Recherche des Spaces HuggingFace (démos déployées).
        Utile pour trouver des demos similaires au projet soumis.
        """
        r = _safe_get(
            f"{self.BASE}/spaces",
            _HF_HEADERS,
            {
                "search":    query,
                "sort":      "likes",
                "direction": -1,
                "limit":     min(max_results * 2, 20),
                "full":      True,
            },
        )
        if r is None:
            return []

        try:
            items = r.json()
            if not isinstance(items, list):
                return []
        except Exception:
            return []

        results = []
        for item in items:
            space_id = item.get("id", "")
            card     = item.get("cardData") or {}
            desc     = (card.get("description") or "").strip()
            if not desc or len(desc) < 10:
                continue

            runtime = (item.get("runtime") or {}).get("stage", "")
            is_live = runtime in ("RUNNING", "RUNNING_BUILDING")

            rel = _relevance_score(query, f"{space_id} {desc}")
            results.append(ExternalResult(
                title       = space_id.split("/")[-1] if "/" in space_id else space_id,
                description = desc[:300],
                url         = f"https://huggingface.co/spaces/{space_id}",
                source      = "huggingface_space",
                author      = item.get("author", ""),
                stars       = item.get("likes", 0) or 0,
                has_paper   = False,
                domain      = "deployed_demo" if is_live else "",
                tags        = item.get("tags", [])[:8],
                published_at= (item.get("createdAt") or "")[:10],
                relevance   = rel,
            ))

        results.sort(key=lambda x: (-x.relevance, -x.stars))
        return results[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# ArXiv Search
# ─────────────────────────────────────────────────────────────────────────────

class ArXivSearcher:
    BASE = "http://export.arxiv.org/api/query"

    # Catégories IA pertinentes
    CATS = ["cs.LG", "cs.CV", "cs.CL", "cs.AI", "cs.RO", "stat.ML", "eess.IV"]

    def search(self, query: str, max_results: int = 8,
               category: str = None) -> list[ExternalResult]:
        """
        Recherche des papers ArXiv récents sur le sujet.
        Si category est fourni (ex: "cs.CV"), recherche dans cette catégorie.
        Sinon cherche dans toutes les catégories IA.
        """
        # Construction de la query ArXiv
        if category:
            search_query = f"cat:{category} AND all:{query}"
        else:
            # Recherche dans le texte libre, tous les champs
            search_query = f"all:{query}"

        r = _safe_get(
            self.BASE,
            {},
            {
                "search_query": search_query,
                "sortBy":       "relevance",
                "sortOrder":    "descending",
                "start":        0,
                "max_results":  min(max_results * 2, 20),
            },
        )
        if r is None:
            log.warning("ArXiv — inaccessible")
            return []

        feed    = feedparser.parse(r.text)
        entries = feed.get("entries", [])

        results = []
        for entry in entries:
            arxiv_url = entry.get("id", "")
            arxiv_id  = arxiv_url.split("/abs/")[-1].split("v")[0].strip()
            title     = entry.get("title", "").replace("\n", " ").strip()
            abstract  = entry.get("summary", "").replace("\n", " ").strip()

            if not abstract or len(abstract) < 40:
                continue

            authors   = entry.get("authors", [])
            author    = authors[0].get("name", "") if authors else ""
            published = entry.get("published", "")[:10]
            rel       = _relevance_score(query, f"{title} {abstract}")

            results.append(ExternalResult(
                title       = title[:200],
                description = abstract[:400],
                url         = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else arxiv_url,
                source      = "arxiv",
                author      = author,
                stars       = 0,
                has_paper   = True,
                paper_url   = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                tags        = [t.get("term", "") for t in entry.get("tags", [])
                               if not t.get("term","").startswith("http")][:6],
                published_at= published,
                relevance   = rel,
            ))

        results.sort(key=lambda x: -x.relevance)
        return results[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Scholar — API publique, aucune clé, passe les proxies
# 200M+ papers académiques. Meilleure alternative à ArXiv si bloqué.
# https://api.semanticscholar.org/graph/v1
# ─────────────────────────────────────────────────────────────────────────────

class SemanticScholarSearcher:
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,authors,year,externalIds,openAccessPdf,citationCount,fieldsOfStudy"

    def search(self, query: str, max_results: int = 8) -> list[ExternalResult]:
        r = _safe_get(
            f"{self.BASE}/paper/search",
            {"Accept": "application/json"},
            {
                "query":  query,
                "limit":  min(max_results * 2, 20),
                "fields": self.FIELDS,
            },
            timeout=15,
        )
        if r is None:
            log.warning("Semantic Scholar — inaccessible")
            return []
        try:
            items = r.json().get("data", [])
        except Exception:
            return []

        results = []
        for item in items:
            title    = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            if not title:
                continue

            # URL : préférer PDF open-access, sinon page S2
            paper_id = item.get("paperId", "")
            pdf      = (item.get("openAccessPdf") or {}).get("url", "")
            arxiv_id = (item.get("externalIds") or {}).get("ArXiv", "")
            url      = pdf or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                               else f"https://www.semanticscholar.org/paper/{paper_id}")

            authors  = item.get("authors", [])
            author   = authors[0].get("name", "") if authors else ""
            desc     = abstract[:400] if abstract else title
            rel      = _relevance_score(query, f"{title} {abstract}")
            citations= item.get("citationCount", 0) or 0

            results.append(ExternalResult(
                title       = title[:200],
                description = desc,
                url         = url,
                source      = "semantic_scholar",
                author      = author,
                stars       = citations,       # on utilise citations comme popularité
                has_paper   = True,
                paper_url   = url,
                tags        = item.get("fieldsOfStudy") or [],
                published_at= str(item.get("year", "")) or "",
                relevance   = rel,
            ))

        results.sort(key=lambda x: (-x.relevance, -x.stars))
        return results[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# Papers With Code miroir — passe les proxies via .co
# ─────────────────────────────────────────────────────────────────────────────

class PapersWithCodeSearcher:
    BASES = [
        "https://paperswithcode.com/api/v1",
        "https://paperswithcode.co/api/v1",
    ]

    def _get_base(self) -> str:
        for base in self.BASES:
            r = _safe_get(f"{base}/papers/", {"Accept": "application/json"},
                          {"items_per_page": 1}, timeout=8)
            if r and r.text:
                try:
                    if "results" in r.json():
                        return base
                except Exception:
                    pass
        return ""

    def search(self, query: str, max_results: int = 8) -> list[ExternalResult]:
        base = self._get_base()
        if not base:
            log.warning("PapersWithCode — inaccessible")
            return []

        r = _safe_get(
            f"{base}/papers/",
            {"Accept": "application/json"},
            {"q": query, "items_per_page": min(max_results * 2, 20)},
        )
        if r is None:
            return []
        try:
            items = r.json().get("results", [])
        except Exception:
            return []

        results = []
        for item in items:
            title  = (item.get("title") or "").strip()
            desc   = (item.get("abstract") or "").strip()
            if not title:
                continue
            arxiv_id = item.get("arxiv_id") or ""
            repos    = item.get("repositories") or []
            gh_url   = repos[0].get("url", "") if repos else ""
            stars    = repos[0].get("stars", 0) if repos else 0
            url      = gh_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                                  else f"https://paperswithcode.com/paper/{item.get('id','')}")
            rel      = _relevance_score(query, f"{title} {desc}")

            results.append(ExternalResult(
                title       = title[:200],
                description = (desc or title)[:400],
                url         = url,
                source      = "pwc",
                author      = "",
                stars       = stars,
                has_paper   = True,
                paper_url   = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url,
                relevance   = rel,
            ))

        results.sort(key=lambda x: (-x.relevance, -x.stars))
        return results[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# Moteur principal — orchestration de toutes les sources
# ─────────────────────────────────────────────────────────────────────────────

class ExternalSearchEngine:
    """
    Moteur de recherche externe avec fallback intelligent.

    Ordre de priorité (selon accessibilité réseau) :
      1. HuggingFace Models   — toujours accessible
      2. HuggingFace Spaces   — toujours accessible
      3. Semantic Scholar     — API publique, passe les proxies
      4. Papers With Code     — .com ou .co selon proxy
      5. GitHub               — optionnel (bloqué sur certains proxies)
      6. ArXiv                — optionnel (bloqué sur certains proxies)
    """

    def __init__(
        self,
        trigger_score:   float = DEFAULT_TRIGGER_SCORE,
        max_per_source:  int   = 8,
        include_arxiv:   bool  = True,
        include_spaces:  bool  = True,
        include_s2:      bool  = True,
        include_pwc:     bool  = True,
    ):
        self.trigger_score  = trigger_score
        self.max_per_source = max_per_source
        self.include_arxiv  = include_arxiv
        self.include_spaces = include_spaces
        self.include_s2     = include_s2
        self.include_pwc    = include_pwc

        self.github = GitHubSearcher()
        self.hf     = HuggingFaceSearcher()
        self.s2     = SemanticScholarSearcher() if include_s2 else None
        self.pwc    = PapersWithCodeSearcher()  if include_pwc else None
        self.arxiv_src = ArXivSearcher()        if include_arxiv else None

    def should_trigger(self, best_internal_score: float) -> bool:
        return best_internal_score < self.trigger_score

    def search(
        self,
        query:        str,
        domain:       str = "",
        tags:         str = "",
        technologies: str = "",
        triggered_by: str = "low_similarity",
        max_total:    int = 15,
    ) -> ExternalSearchResult:

        enriched = query
        if tags:         enriched += " " + tags.replace(";", " ")
        if technologies: enriched += " " + technologies.replace(";", " ")

        log.info(f"Recherche externe déclenchée ({triggered_by}): '{query[:60]}'")

        sources_used: list[str]        = []
        all_results:  list[ExternalResult] = []

        # ── 1. HuggingFace Models (toujours accessible) ───────────────────
        try:
            hf_results = self.hf.search_models(enriched, max_results=self.max_per_source)
            if hf_results:
                all_results.extend(hf_results)
                sources_used.append("huggingface")
                log.info(f"  HuggingFace models → {len(hf_results)}")
        except Exception as e:
            log.debug(f"HF models erreur: {e}")

        # ── 2. HuggingFace Spaces (démos déployées) ───────────────────────
        if self.include_spaces:
            try:
                sp_results = self.hf.search_spaces(
                    enriched, max_results=self.max_per_source // 2)
                if sp_results:
                    all_results.extend(sp_results)
                    if "huggingface" not in sources_used:
                        sources_used.append("huggingface")
                    log.info(f"  HF Spaces → {len(sp_results)}")
            except Exception as e:
                log.debug(f"HF spaces erreur: {e}")

        # ── 3. Semantic Scholar (proxy-safe, papers académiques) ──────────
        if self.include_s2 and self.s2:
            try:
                s2_results = self.s2.search(query, max_results=self.max_per_source)
                if s2_results:
                    all_results.extend(s2_results)
                    sources_used.append("semantic_scholar")
                    log.info(f"  Semantic Scholar → {len(s2_results)}")
            except Exception as e:
                log.debug(f"Semantic Scholar erreur: {e}")

        # ── 4. Papers With Code (.com ou .co) ────────────────────────────
        if self.include_pwc and self.pwc:
            try:
                pwc_results = self.pwc.search(query, max_results=self.max_per_source)
                if pwc_results:
                    all_results.extend(pwc_results)
                    sources_used.append("pwc")
                    log.info(f"  PapersWithCode → {len(pwc_results)}")
            except Exception as e:
                log.debug(f"PWC erreur: {e}")

        # ── 5. GitHub (optionnel) ─────────────────────────────────────────
        try:
            gh_results = self.github.search(enriched, max_results=self.max_per_source)
            if gh_results:
                all_results.extend(gh_results)
                sources_used.append("github")
                log.info(f"  GitHub → {len(gh_results)}")
        except Exception as e:
            log.debug(f"GitHub non accessible: {e}")

        # ── 6. ArXiv (optionnel) ──────────────────────────────────────────
        if self.include_arxiv and self.arxiv_src:
            try:
                category = _domain_to_arxiv_cat(domain)
                ar_results = self.arxiv_src.search(
                    query, max_results=self.max_per_source, category=category)
                if ar_results:
                    all_results.extend(ar_results)
                    sources_used.append("arxiv")
                    log.info(f"  ArXiv → {len(ar_results)}")
            except Exception as e:
                log.debug(f"ArXiv non accessible: {e}")

        # ── Fusion + déduplication + re-ranking ──────────────────────────
        merged    = _deduplicate(all_results)
        max_stars = max((r.stars for r in merged), default=1) or 1
        for r in merged:
            pop_bonus  = min(r.stars / max_stars, 1.0) * 0.15
            r.relevance = round(r.relevance * 0.85 + pop_bonus, 4)

        merged.sort(key=lambda x: -x.relevance)
        merged    = merged[:max_total]
        best_rel  = merged[0].relevance if merged else 0.0

        log.info(
            f"  Total : {len(merged)} résultats fusionnés "
            f"({', '.join(sources_used) or 'aucune source'}) — rel={best_rel:.3f}"
        )

        return ExternalSearchResult(
            query          = query,
            results        = merged,
            sources_used   = sources_used,
            total_found    = len(all_results),
            triggered_by   = triggered_by,
            best_relevance = best_rel,
        )

    def format(self, result: ExternalSearchResult, top_k: int = 10) -> str:
        lines = [
            f"\n🌐 Recherche externe — '{result.query[:60]}'",
            f"   Sources : {', '.join(result.sources_used) or 'aucune'} | "
            f"Trouvés : {result.total_found} → fusionnés : {len(result.results)}",
            "─" * 65,
        ]
        icons = {"github":"⭐","huggingface":"🤗","huggingface_space":"▷",
                 "arxiv":"§","semantic_scholar":"📚","pwc":"◈"}
        for r in result.results[:top_k]:
            icon  = icons.get(r.source, "🔗")
            paper = " [paper]" if r.has_paper else ""
            lines += [
                f"{icon} [{r.relevance:.3f}] {r.title[:50]:<50} ★{r.stars:>6}{paper}",
                f"   {r.description[:100]}...",
                f"   {r.url}", "",
            ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _domain_to_arxiv_cat(domain: str) -> Optional[str]:
    """Mappe un domaine de notre dataset vers une catégorie ArXiv."""
    mapping = {
        "Computer Vision":      "cs.CV",
        "NLP / Text":           "cs.CL",
        "Audio / Speech":       "cs.SD",
        "Generative AI":        "cs.LG",
        "Multimodal":           "cs.CV",
        "Medical / Healthcare": "eess.IV",
        "Robotics / RL":        "cs.RO",
        "Graph / Network":      "cs.LG",
        "3D / Point Cloud":     "cs.CV",
        "Time Series":          "stat.ML",
        "Finance":              "stat.ML",
    }
    return mapping.get(domain)


def _deduplicate(results: list[ExternalResult]) -> list[ExternalResult]:
    """Déduplique par URL et par titre similaire."""
    seen_urls:   set[str] = set()
    seen_titles: set[str] = set()
    out: list[ExternalResult] = []

    for r in results:
        url_key   = r.url.lower().rstrip("/")
        title_key = re.sub(r'\W+', '', r.title.lower())[:40]

        if url_key in seen_urls or title_key in seen_titles:
            continue

        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        out.append(r)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Intégration avec similarity_light.py
# ─────────────────────────────────────────────────────────────────────────────

def search_with_fallback(
    query: str,
    similarity_engine,
    external_engine:      "ExternalSearchEngine" = None,
    tags: str             = "",
    technologies: str     = "",
    trigger_score: float  = DEFAULT_TRIGGER_SCORE,
) -> dict:
    """
    Fonction d'orchestration principale :
      1. Lance la recherche interne via similarity_engine
      2. Si le meilleur score < trigger_score → lance aussi la recherche externe
      3. Retourne un dict avec les deux résultats et la décision prise

    Usage :
        from similarity_light import SimilarityEngineLight
        from external_search import ExternalSearchEngine, search_with_fallback

        sim = SimilarityEngineLight()
        sim.load("ai_projects.csv", "xgboost_bundle.pkl")

        ext = ExternalSearchEngine()

        result = search_with_fallback(
            "brain tumor detection MRI CNN",
            similarity_engine=sim,
            external_engine=ext,
            tags="medical-imaging; segmentation",
            technologies="PyTorch",
        )
        print(result["summary"])
    """
    # Recherche interne
    internal = similarity_engine.search(query, tags=tags, technologies=technologies)
    best_score = internal.similar_projects[0].score if internal.similar_projects else 0.0

    triggered = external_engine is not None and best_score < trigger_score
    external  = None

    if triggered:
        log.info(
            f"Score interne max={best_score:.3f} < seuil={trigger_score} → "
            "déclenchement recherche externe"
        )
        external = external_engine.search(
            query,
            domain       = internal.predicted_domain,
            tags         = tags,
            technologies = technologies,
            triggered_by = "low_similarity",
        )

    summary_lines = [
        f"Query : {query[:80]}",
        f"Domaine prédit : {internal.predicted_domain} "
        f"({internal.domain_confidence:.0%})",
        f"Score interne max : {best_score:.3f} "
        f"({'externe déclenché' if triggered else 'OK'})",
        f"Originalité : {internal.originality_score:.3f} — "
        f"{internal.originality_label}",
        "",
        "── Résultats internes ──",
    ]
    for p in internal.similar_projects[:5]:
        summary_lines.append(
            f"  [{p.score:.3f}] {'✅' if p.same_domain else '↗'} "
            f"{p.name} ({p.domain})"
        )

    if external:
        summary_lines += ["", f"── Résultats externes ({len(external.results)}) ──"]
        for r in external.results[:5]:
            icon = {"github":"⭐","huggingface":"🤗","arxiv":"📄"}.get(r.source,"🔗")
            summary_lines.append(
                f"  {icon} [{r.relevance:.3f}] {r.title[:50]} ★{r.stars}"
            )

    summary_lines += ["", *internal.suggestions]

    return {
        "internal":           internal,
        "external":           external,
        "best_internal_score":best_score,
        "external_triggered": triggered,
        "summary":            "\n".join(summary_lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="External Search Engine")
    parser.add_argument("query",      help="Description du projet")
    parser.add_argument("--domain",   default="", help="Domaine IA (ex: 'Computer Vision')")
    parser.add_argument("--tags",     default="")
    parser.add_argument("--tech",     default="")
    parser.add_argument("--top-k",   type=int, default=10)
    parser.add_argument("--no-arxiv",  action="store_true")
    parser.add_argument("--no-spaces", action="store_true")
    args = parser.parse_args()

    engine = ExternalSearchEngine(
        include_arxiv  = not args.no_arxiv,
        include_spaces = not args.no_spaces,
    )
    result = engine.search(
        args.query,
        domain       = args.domain,
        tags         = args.tags,
        technologies = args.tech,
        triggered_by = "manual",
    )
    print(engine.format(result, top_k=args.top_k))


if __name__ == "__main__":
    _cli()