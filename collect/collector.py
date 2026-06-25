"""
collector.py — Collecte incrémentale et paginée de projets IA
Sources : GitHub REST API v3, HuggingFace Hub API, RSS/Atom (feedparser)

Garanties :
  - Aucune perte de données existantes (INSERT OR IGNORE via fingerprint unique)
  - Reprise sans re-collecter ce qui existe déjà
  - Checkpoint par requête/topic → relance possible sans repartir de zéro
  - Diff détaillé : nouveaux / mis à jour / ignorés
  - Pagination complète (GitHub jusqu'à 1000, HF curseur offset)
"""

import os
import re
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterator

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from models import AIProject, db_session, init_db
from utils import extract_tags_from_text, detect_technologies, score_popularity, extract_paper_info, infer_domain

load_dotenv()
log = logging.getLogger("collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# Tokens & headers
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}
HF_HEADERS = {
    "Accept": "application/json",
    **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
}

REQUEST_DELAY   = 1.0
CHECKPOINT_FILE = Path("collector_checkpoint.json")

# Proxy d'entreprise / antivirus qui intercepte HTTPS → certificat self-signed
# Mettre à False si tu vois : SSLCertVerificationError / self-signed certificate
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() != "false"
if not SSL_VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    log.warning("SSL_VERIFY=false — vérification des certificats désactivée (proxy détecté)")

# ─────────────────────────────────────────────────────────────────────────────
# Requêtes GitHub — 50 queries x pagination = volume maximum
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_QUERIES: list[tuple[str, str]] = [
    ("deep learning medical imaging",            "deep-learning"),
    ("computer vision transformer pytorch",      "computer-vision"),
    ("NLP text classification BERT",             "nlp"),
    ("reinforcement learning robotics",          "reinforcement-learning"),
    ("generative model diffusion stable",        "generative-ai"),
    ("speech recognition wav2vec transformer",   "speech-recognition"),
    ("object detection YOLO real-time",          "object-detection"),
    ("anomaly detection autoencoder unsupervised","anomaly-detection"),
    ("time series forecasting LSTM transformer", "time-series"),
    ("LLM fine-tuning LoRA PEFT",                "llm"),
    ("image segmentation semantic pytorch",      "segmentation"),
    ("graph neural network GNN node",            "graph-neural-network"),
    ("drug discovery molecular protein",         "drug-discovery"),
    ("autonomous driving perception lidar",      "autonomous-driving"),
    ("point cloud 3D reconstruction NeRF",       "3d"),
    ("multimodal vision language CLIP",          "multimodal"),
    ("recommendation system collaborative",      "recommendation-system"),
    ("fraud detection financial tabular",        "fraud-detection"),
    ("satellite remote sensing classification",  "remote-sensing"),
    ("knowledge graph embedding entity",         "knowledge-graph"),
    ("zero-shot few-shot learning meta",         "meta-learning"),
    ("contrastive self-supervised representation","self-supervised"),
    ("neural architecture search NAS",           "neural-architecture-search"),
    ("federated learning privacy distributed",   "federated-learning"),
    ("explainable AI interpretability SHAP",     "explainable-ai"),
    ("text to image generation DALL-E",          "text-to-image"),
    ("video understanding action recognition",   "video-understanding"),
    ("question answering reading comprehension", "question-answering"),
    ("clinical notes EHR healthcare NLP",        "healthcare"),
    ("face recognition verification landmark",   "face-recognition"),
    ("optical flow depth estimation stereo",     "depth-estimation"),
    ("document understanding OCR layout",        "document-ai"),
    ("code generation programming LLM",          "code-generation"),
    ("music generation audio synthesis",         "audio-generation"),
    ("pose estimation human body keypoint",      "pose-estimation"),
    ("crowd counting density map estimation",    "crowd-analysis"),
    ("image restoration super resolution",       "image-restoration"),
    ("chatbot dialogue conversational AI",       "chatbot"),
    ("tabular data gradient boosting XGBoost",   "tabular"),
    ("transfer learning domain adaptation",      "transfer-learning"),
    ("model compression pruning quantization",   "model-compression"),
    ("multilingual cross-lingual NLP",           "multilingual"),
    ("climate weather forecasting prediction",   "climate"),
    ("protein structure prediction alphafold",   "bioinformatics"),
    ("scene graph visual relationship detection","scene-understanding"),
    ("sign language gesture recognition",        "gesture-recognition"),
    ("synthetic data augmentation generation",   "data-augmentation"),
    ("continual learning catastrophic forgetting","continual-learning"),
    ("text summarization abstractive extractive","summarization"),
    ("named entity recognition NER tagging",     "ner"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Topics HuggingFace — tous les pipelines
# ─────────────────────────────────────────────────────────────────────────────

HF_MODEL_TOPICS: list[str] = [
    "image-classification", "object-detection", "image-segmentation",
    "image-to-image", "unconditional-image-generation",
    "text-to-image", "depth-estimation", "image-feature-extraction",
    "video-classification", "zero-shot-image-classification",
    "text-classification", "token-classification", "question-answering",
    "text-generation", "text2text-generation", "summarization",
    "translation", "fill-mask", "sentence-similarity",
    "feature-extraction", "zero-shot-classification",
    "automatic-speech-recognition", "audio-classification",
    "text-to-speech", "audio-to-audio", "voice-activity-detection",
    "visual-question-answering", "document-question-answering",
    "image-text-to-text", "video-text-to-text",
    "tabular-classification", "tabular-regression",
    "reinforcement-learning", "robotics",
]

HF_SPACE_TOPICS: list[str] = [
    "image-classification", "object-detection", "text-generation",
    "automatic-speech-recognition", "text-to-image", "question-answering",
    "summarization", "translation", "image-segmentation",
    "audio-classification", "text-to-speech", "visual-question-answering",
    "reinforcement-learning", "multimodal",
]

HF_RSS_FEEDS: list[str] = [
    "https://huggingface.co/papers.rss",
    "https://huggingface.co/blog.rss",
]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint — progression persistante
# ─────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, path: Path = CHECKPOINT_FILE):
        self.path  = path
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"github": [], "hf_models": [], "hf_spaces": [], "hf_rss": []}

    def save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_done(self, source: str, key: str) -> bool:
        return key in self._data.get(source, [])

    def mark_done(self, source: str, key: str):
        self._data.setdefault(source, [])
        if key not in self._data[source]:
            self._data[source].append(key)
        self.save()

    def reset(self, source: str | None = None):
        if source:
            self._data[source] = []
        else:
            self._data = {"github": [], "hf_models": [], "hf_spaces": [], "hf_rss": []}
        self.save()

    def summary(self) -> dict:
        return {k: len(v) for k, v in self._data.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Compteurs de diff
# ─────────────────────────────────────────────────────────────────────────────

class DiffCounter:
    def __init__(self, source: str):
        self.source   = source
        self.new      = 0
        self.updated  = 0
        self.skipped  = 0
        self.rejected = 0

    def report(self) -> str:
        total = self.new + self.updated + self.skipped
        return (
            f"[{self.source:<12}] traités={total:>5} | "
            f"✅ new={self.new:>4} | "
            f"🔄 updated={self.updated:>4} | "
            f"⏭  skipped={self.skipped:>5} | "
            f"🚫 rejected={self.rejected:>4}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fingerprint(source: str, source_id: str) -> str:
    return hashlib.sha256(
        f"{source}::{source_id}".lower().strip().encode()
    ).hexdigest()


def _safe_date(raw) -> Optional[str]:
    if not raw:
        return None
    # struct_time de feedparser
    if hasattr(raw, "__len__") and not isinstance(raw, str) and len(raw) >= 6:
        try:
            return datetime(*raw[:6], tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    s = str(raw)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d",
                "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
        try:
            return datetime.strptime(s[:29], fmt).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return s[:32]


def _upsert(session, project: AIProject, diff: DiffCounter):
    existing = session.query(AIProject).filter_by(
        fingerprint=project.fingerprint
    ).first()

    if existing is None:
        session.add(project)
        diff.new += 1
        return

    changed = False
    for field in ("stars", "forks", "watchers", "downloads", "open_issues",
                  "is_deployed", "has_demo", "demo_url", "updated_at",
                  "popularity_score", "is_archived"):
        nv = getattr(project, field, None)
        ov = getattr(existing, field, None)
        if nv is not None and nv != ov:
            setattr(existing, field, nv)
            changed = True

    if changed:
        diff.updated += 1
    else:
        diff.skipped += 1


def _safe_get(url: str, headers: dict, params: dict = None,
              retries: int = 3) -> Optional[requests.Response]:
    NO_RETRY_CODES = {400, 401, 403, 404, 410, 412, 422, 451}

    for attempt in range(retries):
        try:
            r = requests.get(
                url, headers=headers, params=params or {},
                timeout=12,           # réduit de 20s → 12s
                verify=SSL_VERIFY,    # bypass proxy SSL si nécessaire
            )
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                log.warning(f"Rate limit — attente {wait}s")
                time.sleep(wait)
                continue
            if r.status_code in NO_RETRY_CODES:
                log.debug(f"HTTP {r.status_code} (no retry): {url}")
                return None
            r.raise_for_status()
            # Proxy intercepteur renvoie parfois du HTML au lieu du JSON attendu
            ct = r.headers.get("Content-Type", "")
            if "text/html" in ct and "json" not in ct:
                log.debug(f"Réponse HTML reçue au lieu de JSON (proxy?) — {url}")
                return None
            return r

        except requests.exceptions.SSLError:
            # Certificat SSL invalide (proxy d'entreprise) — pas la peine de retry
            log.debug(f"SSL error (no retry) — ajoute SSL_VERIFY=false dans .env : {url}")
            return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < retries - 1:
                wait = REQUEST_DELAY * (attempt + 1)
                log.debug(f"Connexion {attempt+1}/{retries} — retry dans {wait:.0f}s: {e}")
                time.sleep(wait)
            else:
                log.debug(f"Échec définitif (ignoré): {url}")
        except requests.RequestException as e:
            log.warning(f"Tentative {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))

    return None


# ─────────────────────────────────────────────────────────────────────────────
# GitHub Collector
# ─────────────────────────────────────────────────────────────────────────────

class GitHubCollector:
    BASE      = "https://api.github.com"
    PAGE_SIZE = 100
    MAX_PAGES = 10   # 10 × 100 = 1000 max par requête (limite GitHub)

    def _paginate(self, query: str, topic: str) -> Iterator[dict]:
        q = f"{query} topic:{topic} language:Python"
        for page in range(1, self.MAX_PAGES + 1):
            r = _safe_get(
                f"{self.BASE}/search/repositories",
                GITHUB_HEADERS,
                {"q": q, "sort": "stars", "order": "desc",
                 "per_page": self.PAGE_SIZE, "page": page},
            )
            if r is None:
                break
            data  = r.json()
            items = data.get("items", [])
            if not items:
                break
            log.info(f"  GitHub p{page} '{query[:35]}' → {len(items)} repos")
            yield from items
            if len(items) < self.PAGE_SIZE:
                break
            time.sleep(REQUEST_DELAY)

    def _readme(self, full_name: str) -> str:
        r = _safe_get(
            f"{self.BASE}/repos/{full_name}/readme",
            {**GITHUB_HEADERS, "Accept": "application/vnd.github.raw"},
        )
        return r.text[:3000] if r else ""

    @staticmethod
    def _deployed(repo: dict) -> bool:
        hp     = repo.get("homepage") or ""
        topics = repo.get("topics", [])
        return hp.startswith("http") or any(
            t in topics for t in
            ("docker","huggingface-spaces","demo","api","streamlit",
             "gradio","fastapi","heroku","aws","gcp","azure","vercel")
        )

    def _parse(self, raw: dict, session, diff: DiffCounter):
        desc = (raw.get("description") or "").strip()
        if not desc:
            diff.rejected += 1
            return
        full_name = raw.get("full_name", "")
        # On n'appelle PAS _readme() — trop lent (1 req réseau/repo, timeouts fréquents)
        # La description + topics suffisent pour la classification et la similarité
        topics    = raw.get("topics", [])
        full_text = f"{desc} {' '.join(topics)}"
        tags      = list(set(topics + extract_tags_from_text(full_text)))
        has_paper, paper_url = extract_paper_info(full_text, tags)
        domain    = infer_domain(desc, tags)
        project = AIProject(
            fingerprint      = _fingerprint("github", full_name),
            source           = "github",
            source_id        = str(raw.get("id", "")),
            name             = raw.get("name", ""),
            full_name        = full_name,
            description      = desc[:2000],
            readme_excerpt   = None,   # désactivé — trop de timeouts réseau
            author           = (raw.get("owner") or {}).get("login", ""),
            url              = raw.get("html_url", ""),
            tags             = json.dumps(tags),
            technologies     = json.dumps(detect_technologies(full_text)),
            language         = raw.get("language") or "",
            stars            = raw.get("stargazers_count", 0),
            forks            = raw.get("forks_count", 0),
            watchers         = raw.get("watchers_count", 0),
            open_issues      = raw.get("open_issues_count", 0),
            license          = (raw.get("license") or {}).get("spdx_id"),
            is_fork          = raw.get("fork", False),
            is_archived      = raw.get("archived", False),
            is_deployed      = self._deployed(raw),
            has_demo         = bool(raw.get("homepage")),
            demo_url         = raw.get("homepage") or None,
            has_paper        = has_paper,
            paper_url        = paper_url,
            domain           = domain,
            popularity_score = score_popularity(
                stars    = raw.get("stargazers_count", 0),
                forks    = raw.get("forks_count", 0),
                watchers = raw.get("watchers_count", 0),
            ),
            published_at     = _safe_date(raw.get("created_at")),
            updated_at       = _safe_date(raw.get("updated_at")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect(self, session, checkpoint: Checkpoint, diff: DiffCounter):
        for query, topic in GITHUB_QUERIES:
            key = f"{query}::{topic}"
            if checkpoint.is_done("github", key):
                log.info(f"  ⏭  GitHub skip: {query[:45]}")
                continue
            batch = 0
            for raw in self._paginate(query, topic):
                self._parse(raw, session, diff)
                batch += 1
                if batch % 50 == 0:
                    session.commit()
            session.commit()
            checkpoint.mark_done("github", key)
            time.sleep(REQUEST_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace RSS (feedparser)
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceRSSCollector:

    def collect(self, session, checkpoint: Checkpoint, diff: DiffCounter):
        for url in HF_RSS_FEEDS:
            if checkpoint.is_done("hf_rss", url):
                log.info(f"  ⏭  RSS skip: {url}")
                continue
            log.info(f"RSS → {url}")
            feed = feedparser.parse(url)
            for entry in feed.entries:
                self._parse(entry, session, diff)
            session.commit()
            checkpoint.mark_done("hf_rss", url)
            time.sleep(REQUEST_DELAY)

    @staticmethod
    def _parse(entry, session, diff: DiffCounter):
        link = entry.get("link", "")
        raw  = entry.get("summary", "") or entry.get("description", "")
        desc = re.sub(r"\s+", " ",
            BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
        ).strip()
        if not desc or len(desc) < 30:
            diff.rejected += 1
            return
        tags_raw = [t.get("term", "") for t in entry.get("tags", [])]
        project = AIProject(
            fingerprint      = _fingerprint("hf_rss", link),
            source           = "hf_rss",
            source_id        = link,
            name             = entry.get("title", "").strip()[:256],
            full_name        = f"hf_rss::{entry.get('title','')[:80]}",
            description      = desc[:2000],
            author           = entry.get("author", ""),
            url              = link,
            tags             = json.dumps(list(set(
                                   tags_raw + extract_tags_from_text(desc)))),
            technologies     = json.dumps(detect_technologies(desc)),
            is_deployed      = False,
            has_demo         = False,
            stars=0, forks=0, watchers=0, open_issues=0,
            popularity_score = 0.0,
            published_at     = _safe_date(
                entry.get("published_parsed") or entry.get("published")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Hub API (models + spaces) — pagination offset complète
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceHubCollector:
    BASE      = "https://huggingface.co/api"
    PAGE_SIZE = 100

    def _paginate(self, endpoint: str, topic: str, sort: str,
                  limit: int) -> Iterator[dict]:
        offset = 0
        fetched = 0
        while fetched < limit:
            r = _safe_get(f"{self.BASE}/{endpoint}", HF_HEADERS, {
                "filter": topic, "sort": sort, "direction": -1,
                "limit": min(self.PAGE_SIZE, limit - fetched),
                "skip": offset, "cardData": True, "full": True,
            })
            if r is None:
                break
            items = r.json()
            if not isinstance(items, list) or not items:
                break
            log.info(f"  HF {endpoint} [{topic}] skip={offset} → {len(items)}")
            yield from items
            fetched += len(items)
            if len(items) < self.PAGE_SIZE:
                break
            offset += self.PAGE_SIZE
            time.sleep(REQUEST_DELAY)

    @staticmethod
    def _build_description(raw: dict) -> str:
        """
        Construit une description depuis les métadonnées déjà dans la réponse API.
        Aucun appel réseau supplémentaire — évite les torrents de 404 /readme.

        Priorité :
          1. cardData.description (model card explicite)
          2. raw.description
          3. Synthèse pipeline_tag + tags + author (fallback minimal)
        """
        card = raw.get("cardData") or {}

        desc = (card.get("description") or raw.get("description") or "").strip()
        if desc:
            return desc[:2000]

        # Fallback : synthèse depuis les champs structurés
        model_id = raw.get("modelId") or raw.get("id", "")
        pipeline = raw.get("pipeline_tag") or ""
        tags     = raw.get("tags", []) or []
        author   = raw.get("author") or (model_id.split("/")[0] if "/" in model_id else "")
        name     = model_id.split("/")[-1] if "/" in model_id else model_id

        meaningful_tags = [
            t for t in tags[:10]
            if not any(t.startswith(p) for p in
                       ("license:", "arxiv:", "doi:", "region:", "base_model:"))
        ]

        parts = []
        if name:
            parts.append(name)
        if pipeline:
            parts.append(f"— {pipeline} model")
        if author:
            parts.append(f"by {author}")
        if meaningful_tags:
            parts.append("| " + ", ".join(meaningful_tags))

        synth = " ".join(parts).strip()
        return synth if len(synth) >= 20 else ""

    # ── Models ────────────────────────────────────────────────────────────

    def _parse_model(self, raw: dict, session, diff: DiffCounter):
        model_id = raw.get("modelId") or raw.get("id", "")
        desc     = self._build_description(raw)
        if not desc:
            diff.rejected += 1
            return

        tags_raw = raw.get("tags", []) or []
        pipeline = raw.get("pipeline_tag") or ""
        if pipeline:
            tags_raw = [pipeline] + tags_raw
        likes     = raw.get("likes", 0) or 0
        downloads = raw.get("downloads", 0) or 0
        author    = raw.get("author") or (model_id.split("/")[0] if "/" in model_id else "")
        tags_full = list(set(tags_raw + extract_tags_from_text(desc)))
        has_paper, paper_url = extract_paper_info(desc, tags_raw)
        domain    = infer_domain(desc, tags_full, pipeline_tag=pipeline)

        project = AIProject(
            fingerprint      = _fingerprint("hf_model", model_id),
            source           = "hf_model",
            source_id        = model_id,
            name             = model_id.split("/")[-1] if "/" in model_id else model_id,
            full_name        = model_id,
            description      = desc[:2000],
            author           = author,
            url              = f"https://huggingface.co/{model_id}",
            tags             = json.dumps(tags_full),
            technologies     = json.dumps(
                                   detect_technologies(" ".join(tags_raw) + " " + desc)),
            pipeline_tag     = pipeline,
            stars            = likes,
            forks=0, watchers=0, open_issues=0,
            downloads        = downloads,
            is_deployed      = False,
            has_demo         = False,
            has_paper        = has_paper,
            paper_url        = paper_url,
            domain           = domain,
            popularity_score = score_popularity(stars=likes, downloads=downloads),
            published_at     = _safe_date(raw.get("createdAt") or raw.get("lastModified")),
            updated_at       = _safe_date(raw.get("lastModified")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect_models(self, session, checkpoint: Checkpoint, diff: DiffCounter,
                       max_per_topic: int = 500):
        for topic in HF_MODEL_TOPICS:
            key = f"model::{topic}"
            if checkpoint.is_done("hf_models", key):
                log.info(f"  ⏭  HF model skip: {topic}")
                continue
            count = 0
            for raw in self._paginate("models", topic, "downloads", max_per_topic):
                self._parse_model(raw, session, diff)
                count += 1
                if count % 100 == 0:
                    session.commit()
            session.commit()
            checkpoint.mark_done("hf_models", key)
            log.info(f"  ✔ HF models [{topic}] — {count} traités")
            time.sleep(REQUEST_DELAY)

    # ── Spaces ────────────────────────────────────────────────────────────

    def _parse_space(self, raw: dict, session, diff: DiffCounter):
        space_id = raw.get("id", "")
        card     = raw.get("cardData") or {}
        desc     = (card.get("description") or raw.get("description") or "").strip()
        if not desc or len(desc) < 20:
            diff.rejected += 1
            return

        tags_raw      = raw.get("tags", []) or []
        runtime_stage = (raw.get("runtime") or {}).get("stage", "")
        is_running    = runtime_stage in ("RUNNING", "RUNNING_BUILDING")
        likes         = raw.get("likes", 0) or 0
        author        = raw.get("author") or (space_id.split("/")[0] if "/" in space_id else "")

        project = AIProject(
            fingerprint      = _fingerprint("hf_space", space_id),
            source           = "hf_space",
            source_id        = space_id,
            name             = space_id.split("/")[-1] if "/" in space_id else space_id,
            full_name        = space_id,
            description      = desc[:2000],
            author           = author,
            url              = f"https://huggingface.co/spaces/{space_id}",
            tags             = json.dumps(list(set(
                                   tags_raw + extract_tags_from_text(desc)))),
            technologies     = json.dumps(
                                   detect_technologies(" ".join(tags_raw) + " " + desc)),
            stars            = likes,
            forks=0, watchers=0, open_issues=0,
            is_deployed      = is_running,
            has_demo         = is_running,
            demo_url         = f"https://huggingface.co/spaces/{space_id}" if is_running else None,
            popularity_score = score_popularity(stars=likes),
            published_at     = _safe_date(raw.get("createdAt")),
            updated_at       = _safe_date(raw.get("lastModified")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect_spaces(self, session, checkpoint: Checkpoint, diff: DiffCounter,
                       max_per_topic: int = 200):
        for topic in HF_SPACE_TOPICS:
            key = f"space::{topic}"
            if checkpoint.is_done("hf_spaces", key):
                log.info(f"  ⏭  HF space skip: {topic}")
                continue
            count = 0
            for raw in self._paginate("spaces", topic, "likes", max_per_topic):
                self._parse_space(raw, session, diff)
                count += 1
                if count % 100 == 0:
                    session.commit()
            session.commit()
            checkpoint.mark_done("hf_spaces", key)
            log.info(f"  ✔ HF spaces [{topic}] — {count} traités")
            time.sleep(REQUEST_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# Papers With Code Collector
# Essaie paperswithcode.com d'abord, bascule sur paperswithcode.co si bloqué.
# Pagination complète par tâche. Aucune clé requise.
# ─────────────────────────────────────────────────────────────────────────────

PWC_HEADERS = {"Accept": "application/json"}

PWC_BASES = [
    "https://paperswithcode.com/api/v1",   # officiel
    "https://paperswithcode.co/api/v1",    # miroir — passe les proxies d'entreprise
]

PWC_TASKS: list[str] = [
    "image-classification", "object-detection", "semantic-segmentation",
    "image-generation", "image-to-image-translation",
    "text-classification", "named-entity-recognition", "question-answering",
    "machine-translation", "text-generation", "language-modelling",
    "sentiment-analysis", "summarization",
    "speech-recognition", "audio-classification", "text-to-speech",
    "reinforcement-learning", "continuous-control",
    "graph-classification", "link-prediction",
    "medical-image-segmentation", "drug-discovery",
    "time-series-forecasting", "anomaly-detection",
    "3d-object-detection", "depth-estimation",
    "visual-question-answering", "image-captioning",
    "knowledge-graph-completion", "face-recognition",
    "action-recognition", "pose-estimation",
]


class PapersWithCodeCollector:
    PAGE_SIZE = 50

    def _detect_working_base(self) -> str:
        """Trouve quelle URL fonctionne (com ou co) en testant une requête légère."""
        for base in PWC_BASES:
            r = _safe_get(f"{base}/papers/", PWC_HEADERS,
                          {"items_per_page": 1, "page": 1})
            if r is not None and r.text and r.text.strip():
                try:
                    data = r.json()
                    if "results" in data:
                        log.info(f"PWC — base active : {base}")
                        return base
                except Exception:
                    pass
        log.warning("PWC — aucune base accessible (proxy bloque les deux URLs)")
        return ""

    def _paginate_task(self, base: str, task: str,
                       max_items: int) -> Iterator[dict]:
        page    = 1
        fetched = 0
        while fetched < max_items:
            r = _safe_get(f"{base}/papers/", PWC_HEADERS, {
                "task": task, "ordering": "-github_star_count",
                "page": page, "items_per_page": self.PAGE_SIZE,
            })
            if r is None or not r.text or not r.text.strip():
                break
            try:
                data = r.json()
            except Exception:
                log.debug(f"PWC [{task}] JSON invalide p{page}")
                break
            results = data.get("results", [])
            if not results:
                break
            log.info(f"  PWC [{task}] p{page} → {len(results)} papers")
            yield from results
            fetched += len(results)
            if not data.get("next"):
                break
            page += 1
            time.sleep(REQUEST_DELAY)

    def _parse(self, raw: dict, session, diff: DiffCounter):
        paper_id = raw.get("id") or raw.get("arxiv_id") or ""
        if not paper_id:
            diff.rejected += 1
            return

        title = (raw.get("title") or "").strip()
        desc  = (raw.get("abstract") or "").strip()
        if not desc:
            diff.rejected += 1
            return

        repos      = raw.get("repositories") or []
        github_url = repos[0].get("url", "") if repos else ""
        stars      = repos[0].get("stars", 0) if repos else 0
        framework  = repos[0].get("framework", "") if repos else ""
        arxiv_id   = raw.get("arxiv_id") or ""
        paper_url  = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id \
                     else raw.get("url_pdf") or ""

        tags_raw = [t.get("name", "") for t in (raw.get("tasks") or [])]
        methods  = [m.get("name", "") for m in (raw.get("methods") or [])]
        all_text = f"{title} {desc} {' '.join(tags_raw)} {' '.join(methods)} {framework}"

        technologies = detect_technologies(all_text)
        if framework and framework not in technologies:
            technologies.append(framework)

        tags   = list(set(tags_raw + extract_tags_from_text(desc)))
        domain = infer_domain(desc, tags)

        project = AIProject(
            fingerprint      = _fingerprint("pwc", paper_id),
            source           = "pwc",
            source_id        = paper_id,
            name             = title[:256],
            full_name        = f"pwc::{paper_id}",
            description      = desc[:2000],
            author           = "",
            url              = github_url or f"https://paperswithcode.com/paper/{paper_id}",
            tags             = json.dumps(tags),
            technologies     = json.dumps(technologies),
            language         = "Python",
            stars            = stars,
            forks=0, watchers=0, open_issues=0,
            has_paper        = True,
            paper_url        = paper_url or f"https://paperswithcode.com/paper/{paper_id}",
            is_deployed      = False,
            has_demo         = False,
            domain           = domain,
            popularity_score = score_popularity(stars=stars),
            published_at     = _safe_date(raw.get("published") or raw.get("date")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect(self, session, checkpoint: Checkpoint, diff: DiffCounter,
                max_per_task: int = 200):
        base = self._detect_working_base()
        if not base:
            log.warning("PWC — skip complet (aucune URL accessible)")
            return

        for task in PWC_TASKS:
            key = f"pwc::{task}"
            if checkpoint.is_done("pwc", key):
                log.info(f"  ⏭  PWC skip: {task}")
                continue
            count = 0
            for raw in self._paginate_task(base, task, max_per_task):
                self._parse(raw, session, diff)
                count += 1
                if count % 100 == 0:
                    session.commit()
            session.commit()
            checkpoint.mark_done("pwc", key)
            log.info(f"  ✔ PWC [{task}] — {count} papers traités")
            time.sleep(REQUEST_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# ArXiv Collector
# API Atom officielle. Catégories ML/CV/NLP/AI/Stats.
# Aucune clé requise. Respecter le délai de 3s entre requêtes (politique ArXiv).
# ─────────────────────────────────────────────────────────────────────────────

ARXIV_CATEGORIES: list[str] = [
    "cs.LG",   # Machine Learning
    "cs.CV",   # Computer Vision
    "cs.CL",   # Computation and Language (NLP)
    "cs.AI",   # Artificial Intelligence
    "cs.NE",   # Neural and Evolutionary Computing
    "cs.RO",   # Robotics
    "cs.SD",   # Sound / Audio
    "stat.ML", # Machine Learning (stats)
    "eess.IV", # Image and Video Processing
    "eess.AS", # Audio and Speech Processing
    "q-bio.QM",# Quantitative Methods (bio/drug discovery)
]

ARXIV_DELAY = 3.0  # délai obligatoire entre requêtes ArXiv


class ArXivCollector:
    BASE      = "http://export.arxiv.org/api/query"
    PAGE_SIZE = 100

    def _fetch(self, category: str, start: int) -> list[dict]:
        """Fetch une page de papers ArXiv pour une catégorie."""
        r = _safe_get(self.BASE, {}, {
            "search_query": f"cat:{category}",
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
            "start":        start,
            "max_results":  self.PAGE_SIZE,
        })
        if r is None:
            return []
        feed    = feedparser.parse(r.text)
        entries = feed.get("entries", [])
        log.info(f"  ArXiv [{category}] start={start} → {len(entries)} papers")
        return entries

    def _parse(self, entry: dict, session, diff: DiffCounter):
        # ID ArXiv = URL de la forme http://arxiv.org/abs/2301.12345v1
        arxiv_url = entry.get("id", "")
        arxiv_id  = arxiv_url.split("/abs/")[-1].split("v")[0].strip()
        if not arxiv_id:
            diff.rejected += 1
            return

        title = entry.get("title", "").replace("\n", " ").strip()
        desc  = entry.get("summary", "").replace("\n", " ").strip()
        if not desc or len(desc) < 40:
            diff.rejected += 1
            return

        fp = _fingerprint("arxiv", arxiv_id)

        authors_raw = entry.get("authors", [])
        author      = authors_raw[0].get("name", "") if authors_raw else ""

        tags_raw = [t.get("term", "") for t in entry.get("tags", [])
                    if not t.get("term", "").startswith("http")]
        tags     = list(set(tags_raw + extract_tags_from_text(desc)))
        domain   = infer_domain(desc, tags)
        techs    = detect_technologies(desc)

        published = _safe_date(
            entry.get("published_parsed") or entry.get("published")
        )

        project = AIProject(
            fingerprint      = fp,
            source           = "arxiv",
            source_id        = arxiv_id,
            name             = title[:256],
            full_name        = f"arxiv::{arxiv_id}",
            description      = desc[:2000],
            author           = author[:256],
            url              = f"https://arxiv.org/abs/{arxiv_id}",
            tags             = json.dumps(tags),
            technologies     = json.dumps(techs),
            language         = None,  # papers, pas forcément du code
            stars=0, forks=0, watchers=0, open_issues=0,
            has_paper        = True,
            paper_url        = f"https://arxiv.org/abs/{arxiv_id}",
            is_deployed      = False,
            has_demo         = False,
            domain           = domain,
            popularity_score = 0.0,
            published_at     = published,
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect(self, session, checkpoint: Checkpoint, diff: DiffCounter,
                max_per_category: int = 500):
        for cat in ARXIV_CATEGORIES:
            key = f"arxiv::{cat}"
            if checkpoint.is_done("arxiv", key):
                log.info(f"  ⏭  ArXiv skip: {cat}")
                continue
            start   = 0
            count   = 0
            while count < max_per_category:
                entries = self._fetch(cat, start)
                if not entries:
                    break
                for entry in entries:
                    self._parse(entry, session, diff)
                    count += 1
                if count % 100 == 0:
                    session.commit()
                if len(entries) < self.PAGE_SIZE:
                    break
                start += self.PAGE_SIZE
                time.sleep(ARXIV_DELAY)   # délai obligatoire ArXiv
            session.commit()
            checkpoint.mark_done("arxiv", key)
            log.info(f"  ✔ ArXiv [{cat}] — {count} papers traités")
            time.sleep(ARXIV_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle Collector
# API officielle via la bibliothèque kaggle ou requêtes directes.
# Collecte : competitions + notebooks/kernels IA.
# Clé : KAGGLE_USERNAME + KAGGLE_KEY dans .env (gratuit sur kaggle.com/settings)
# ─────────────────────────────────────────────────────────────────────────────

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY", "")

KAGGLE_SEARCH_TERMS: list[str] = [
    "deep learning", "computer vision", "natural language processing",
    "image classification", "object detection", "text classification",
    "time series forecasting", "anomaly detection", "recommendation system",
    "speech recognition", "medical imaging", "fraud detection",
    "reinforcement learning", "generative AI", "transformer",
    "neural network", "machine learning", "data science",
]


class KaggleCollector:
    BASE = "https://www.kaggle.com/api/v1"

    def _headers(self) -> dict:
        import base64
        creds = base64.b64encode(
            f"{KAGGLE_USERNAME}:{KAGGLE_KEY}".encode()
        ).decode()
        return {
            "Accept":        "application/json",
            "Authorization": f"Basic {creds}",
        }

    def _available(self) -> bool:
        if not KAGGLE_USERNAME or not KAGGLE_KEY:
            log.warning("Kaggle — KAGGLE_USERNAME/KAGGLE_KEY absents dans .env — skip")
            return False
        # Test de connectivité avec un seul résultat
        r = _safe_get(f"{self.BASE}/competitions/list", self._headers(),
                      {"page": 1, "pageSize": 1})
        if r is None:
            log.warning("Kaggle — inaccessible (proxy bloque kaggle.com/api) — skip")
            return False
        if r.status_code == 401:
            log.warning("Kaggle — clé invalide (401 Unauthorized) — vérifie KAGGLE_USERNAME/KEY dans .env")
            return False
        log.info(f"Kaggle — connecté ✅ (HTTP {r.status_code})")
        return True

    def _search_competitions(self, search: str, page: int = 1) -> list[dict]:
        r = _safe_get(f"{self.BASE}/competitions/list", self._headers(), {
            "search": search, "page": page, "pageSize": 50,
            "sortBy": "numberOfEntrants",
        })
        if r is None or not r.text:
            return []
        try:
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _search_kernels(self, search: str, page: int = 1) -> list[dict]:
        r = _safe_get(f"{self.BASE}/kernels/list", self._headers(), {
            "search": search, "page": page, "pageSize": 50,
            "sortBy": "voteCount", "language": "python",
        })
        if r is None or not r.text:
            return []
        try:
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _parse_competition(self, raw: dict, session, diff: DiffCounter):
        comp_id = str(raw.get("id") or raw.get("ref") or "")
        if not comp_id:
            diff.rejected += 1
            return
        title = (raw.get("title") or "").strip()
        desc  = (raw.get("description") or raw.get("subtitle") or "").strip()
        if not desc:
            desc = title   # titre seul si pas de description
        if not desc:
            diff.rejected += 1
            return

        tags   = extract_tags_from_text(f"{title} {desc}")
        domain = infer_domain(desc, tags)
        url    = f"https://www.kaggle.com/c/{raw.get('ref', comp_id)}"

        project = AIProject(
            fingerprint      = _fingerprint("kaggle_comp", comp_id),
            source           = "kaggle",
            source_id        = comp_id,
            name             = title[:256],
            full_name        = f"kaggle::{raw.get('ref', comp_id)}",
            description      = desc[:2000],
            author           = raw.get("organizationName") or raw.get("hostSegmentTitle") or "",
            url              = url,
            tags             = json.dumps(tags),
            technologies     = json.dumps(detect_technologies(f"{title} {desc}")),
            language         = "Python",
            stars            = raw.get("totalTeams") or raw.get("numberOfEntrants") or 0,
            forks=0, watchers=0, open_issues=0,
            has_paper        = False,
            is_deployed      = False,
            has_demo         = False,
            domain           = domain,
            popularity_score = score_popularity(
                stars=raw.get("totalTeams") or raw.get("numberOfEntrants") or 0),
            published_at     = _safe_date(raw.get("enabledDate") or raw.get("deadline")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def _parse_kernel(self, raw: dict, session, diff: DiffCounter):
        kernel_id = str(raw.get("id") or "")
        ref       = raw.get("ref") or raw.get("currentRunningVersion", {}).get("sourceUrl", "")
        title     = (raw.get("title") or "").strip()
        desc      = title   # kernels n'ont pas de description dans l'API list
        if not desc:
            diff.rejected += 1
            return

        tags      = extract_tags_from_text(desc)
        domain    = infer_domain(desc, tags)
        author    = (raw.get("author") or "").strip()
        votes     = raw.get("totalVotes") or 0
        url       = f"https://www.kaggle.com/code/{ref}" if ref else \
                    f"https://www.kaggle.com/kernels/{kernel_id}"

        project = AIProject(
            fingerprint      = _fingerprint("kaggle_kernel", kernel_id or ref),
            source           = "kaggle",
            source_id        = kernel_id or ref,
            name             = title[:256],
            full_name        = f"kaggle::kernel::{ref or kernel_id}",
            description      = desc[:2000],
            author           = author,
            url              = url,
            tags             = json.dumps(tags),
            technologies     = json.dumps(detect_technologies(desc)),
            language         = "Python",
            stars            = votes,
            forks=0, watchers=0, open_issues=0,
            has_paper        = False,
            is_deployed      = False,
            has_demo         = True,   # les notebooks sont exécutables
            domain           = domain,
            popularity_score = score_popularity(stars=votes),
            published_at     = _safe_date(raw.get("lastRunTime")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect(self, session, checkpoint: Checkpoint, diff: DiffCounter,
                max_per_term: int = 100):
        if not self._available():
            return

        for term in KAGGLE_SEARCH_TERMS:
            # Compétitions
            key_comp = f"kaggle_comp::{term}"
            if not checkpoint.is_done("kaggle", key_comp):
                page = 1
                count = 0
                while count < max_per_term:
                    items = self._search_competitions(term, page)
                    if not items:
                        break
                    for raw in items:
                        self._parse_competition(raw, session, diff)
                        count += 1
                    session.commit()
                    if len(items) < 50:
                        break
                    page += 1
                    time.sleep(REQUEST_DELAY)
                checkpoint.mark_done("kaggle", key_comp)
                log.info(f"  ✔ Kaggle comp [{term}] — {count} résultats")

            # Notebooks
            key_kern = f"kaggle_kern::{term}"
            if not checkpoint.is_done("kaggle", key_kern):
                page = 1
                count = 0
                while count < max_per_term:
                    items = self._search_kernels(term, page)
                    if not items:
                        break
                    for raw in items:
                        self._parse_kernel(raw, session, diff)
                        count += 1
                    session.commit()
                    if len(items) < 50:
                        break
                    page += 1
                    time.sleep(REQUEST_DELAY)
                checkpoint.mark_done("kaggle", key_kern)
                log.info(f"  ✔ Kaggle kernel [{term}] — {count} résultats")

            time.sleep(REQUEST_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# Zenodo Collector
# Dépôt de recherche ouvert (CERN). ~50 000 dépôts IA/ML.
# API REST publique, aucune clé requise (token optionnel pour +débit).
# https://zenodo.org/api/records
# ─────────────────────────────────────────────────────────────────────────────

ZENODO_TOKEN = os.getenv("ZENODO_TOKEN", "")   # optionnel — augmente le débit

ZENODO_QUERIES: list[str] = [
    "deep learning", "machine learning", "neural network",
    "computer vision", "natural language processing",
    "convolutional neural network", "transformer model",
    "object detection", "image segmentation", "speech recognition",
    "reinforcement learning", "generative adversarial network",
    "medical image analysis", "drug discovery",
    "time series prediction", "anomaly detection",
    "graph neural network", "federated learning",
    "model compression", "explainable AI",
]


class ZenodoCollector:
    BASE      = "https://zenodo.org/api/records"
    PAGE_SIZE = 100

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if ZENODO_TOKEN:
            h["Authorization"] = f"Bearer {ZENODO_TOKEN}"
        return h

    def _available(self) -> bool:
        """Test de connectivité Zenodo."""
        r = _safe_get(self.BASE, self._headers(), {"q": "deep learning", "size": 1})
        if r is None:
            log.warning("Zenodo — inaccessible (proxy bloque zenodo.org) — skip")
            return False
        try:
            data = r.json()
            if "hits" in data:
                log.info("Zenodo — connecté ✅")
                return True
        except Exception:
            pass
        log.warning(f"Zenodo — réponse inattendue: {r.text[:100]}")
        return False

    def _search(self, query: str, page: int) -> tuple[list[dict], int]:
        r = _safe_get(self.BASE, self._headers(), {
            "q":    f"{query} resource_type.type:software",
            "sort": "mostrecent",
            "page": page,
            "size": self.PAGE_SIZE,
        })
        if r is None or not r.text:
            return [], 0
        try:
            data  = r.json()
            hits  = data.get("hits", {})
            items = hits.get("hits", [])
            total = hits.get("total", 0)
            if isinstance(total, dict):
                total = total.get("value", 0)
            return items, total
        except Exception:
            return [], 0

    def _parse(self, raw: dict, session, diff: DiffCounter):
        zenodo_id = str(raw.get("id") or "")
        if not zenodo_id:
            diff.rejected += 1
            return

        meta  = raw.get("metadata", {})
        title = (meta.get("title") or "").strip()
        desc  = BeautifulSoup(
            meta.get("description") or "", "html.parser"
        ).get_text(" ", strip=True).strip()
        desc  = re.sub(r"\s+", " ", desc)

        if not desc or len(desc) < 20:
            desc = title
        if not desc:
            diff.rejected += 1
            return

        # Auteurs
        creators = meta.get("creators", [])
        author   = creators[0].get("name", "") if creators else ""

        # Keywords / tags
        kw_raw   = [k.get("subject", k) if isinstance(k, dict) else str(k)
                    for k in meta.get("keywords", [])]
        tags     = list(set(kw_raw + extract_tags_from_text(f"{title} {desc}")))
        domain   = infer_domain(desc, tags)
        techs    = detect_technologies(f"{title} {desc} {' '.join(kw_raw)}")

        # Popularité : nb de téléchargements ou de vues
        stats    = raw.get("stats", {})
        views    = stats.get("unique_views", 0) or 0
        downloads= stats.get("unique_downloads", 0) or 0

        # DOI → paper si disponible
        doi      = meta.get("doi") or raw.get("doi") or ""
        has_paper= bool(doi)
        paper_url= f"https://doi.org/{doi}" if doi else ""

        project = AIProject(
            fingerprint      = _fingerprint("zenodo", zenodo_id),
            source           = "zenodo",
            source_id        = zenodo_id,
            name             = title[:256],
            full_name        = f"zenodo::{zenodo_id}",
            description      = desc[:2000],
            author           = author[:256],
            url              = raw.get("links", {}).get("html", f"https://zenodo.org/record/{zenodo_id}"),
            tags             = json.dumps(tags),
            technologies     = json.dumps(techs),
            language         = None,
            stars            = downloads,
            forks=0, watchers=0, open_issues=0,
            downloads        = downloads,
            has_paper        = has_paper,
            paper_url        = paper_url or None,
            is_deployed      = False,
            has_demo         = False,
            domain           = domain,
            popularity_score = score_popularity(downloads=downloads, stars=views // 10),
            published_at     = _safe_date(meta.get("publication_date")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect(self, session, checkpoint: Checkpoint, diff: DiffCounter,
                max_per_query: int = 200):
        if not self._available():
            return
        for query in ZENODO_QUERIES:
            key = f"zenodo::{query}"
            if checkpoint.is_done("zenodo", key):
                log.info(f"  ⏭  Zenodo skip: {query}")
                continue

            page    = 1
            count   = 0
            fetched = 0
            while fetched < max_per_query:
                items, total = self._search(query, page)
                if not items:
                    break
                for raw in items:
                    self._parse(raw, session, diff)
                    count += 1
                fetched += len(items)
                session.commit()
                if len(items) < self.PAGE_SIZE or fetched >= total:
                    break
                page += 1
                time.sleep(REQUEST_DELAY)

            checkpoint.mark_done("zenodo", key)
            log.info(f"  ✔ Zenodo [{query}] — {count} résultats")
            time.sleep(REQUEST_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# OpenML Collector
# Plateforme ML ouverte — datasets + flows (pipelines) + runs.
# API REST sans clé. ~4 000 flows IA référencés.
# https://www.openml.org/apis
# ─────────────────────────────────────────────────────────────────────────────

OPENML_TAGS: list[str] = [
    "deep_learning", "classification", "regression", "nlp",
    "computer_vision", "ensemble", "neural_network", "sklearn",
    "pytorch", "tensorflow", "xgboost", "study_14", "study_34",
]


class OpenMLCollector:
    BASE = "https://www.openml.org/api/v1/json"
    # OpenML exige ce header exact — sans lui retourne 412
    HEADERS = {
        "Accept":       "application/json",
        "Content-Type": "application/json",
    }

    def _search_flows(self, tag: str, offset: int = 0) -> list[dict]:
        # L'API OpenML filtre par tag via paramètre query, pas dans le chemin
        r = _safe_get(
            f"{self.BASE}/flow/list",
            self.HEADERS,
            {"tag": tag, "limit": 100, "offset": offset},
        )
        if r is None or not r.text:
            return []
        try:
            data = r.json()
            flows = data.get("flows", {})
            if isinstance(flows, dict):
                return flows.get("flow", [])
            return []
        except Exception:
            return []

    def _parse_flow(self, raw: dict, session, diff: DiffCounter):
        flow_id = str(raw.get("id") or "")
        name    = str(raw.get("name") or "").strip()
        desc    = str(raw.get("description") or name).strip()
        if not desc or len(desc) < 10:
            diff.rejected += 1
            return

        # uploader peut être int (user_id) ou string — on convertit proprement
        uploader_raw = raw.get("uploader") or raw.get("creator") or ""
        author       = str(uploader_raw)[:256]

        tags   = extract_tags_from_text(f"{name} {desc}")
        domain = infer_domain(desc, tags)
        techs  = detect_technologies(f"{name} {desc}")

        project = AIProject(
            fingerprint      = _fingerprint("openml", flow_id),
            source           = "openml",
            source_id        = flow_id,
            name             = name[:256],
            full_name        = f"openml::{flow_id}",
            description      = desc[:2000],
            author           = author,
            url              = f"https://www.openml.org/flow/{flow_id}",
            tags             = json.dumps(tags),
            technologies     = json.dumps(techs),
            language         = "Python",
            stars            = int(raw.get("nr_of_likes") or 0),
            forks=0, watchers=0, open_issues=0,
            downloads        = int(raw.get("nr_of_downloads") or 0),
            has_paper        = False,
            is_deployed      = False,
            has_demo         = False,
            domain           = domain,
            popularity_score = score_popularity(
                stars     = int(raw.get("nr_of_likes") or 0),
                downloads = int(raw.get("nr_of_downloads") or 0),
            ),
            published_at     = _safe_date(raw.get("upload_date")),
            collected_at     = datetime.now(timezone.utc).isoformat(),
        )
        _upsert(session, project, diff)

    def collect(self, session, checkpoint: Checkpoint, diff: DiffCounter,
                max_per_tag: int = 500):
        for tag in OPENML_TAGS:
            key = f"openml::{tag}"
            if checkpoint.is_done("openml", key):
                log.info(f"  ⏭  OpenML skip: {tag}")
                continue

            offset = 0
            count  = 0
            while count < max_per_tag:
                items = self._search_flows(tag, offset)
                if not items:
                    break
                for raw in items:
                    self._parse_flow(raw, session, diff)
                    count += 1
                session.commit()
                if len(items) < 100:
                    break
                offset += 100
                time.sleep(REQUEST_DELAY)

            checkpoint.mark_done("openml", key)
            log.info(f"  ✔ OpenML [{tag}] — {count} flows")
            time.sleep(REQUEST_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# Mise à jour Checkpoint — migration propre pour toutes les sources
# ─────────────────────────────────────────────────────────────────────────────

_original_checkpoint_load = Checkpoint._load

def _new_checkpoint_load(self):
    data = _original_checkpoint_load(self)
    for key in ("pwc", "arxiv", "kaggle", "zenodo", "openml"):
        if key not in data:
            data[key] = []
    return data

Checkpoint._load = _new_checkpoint_load


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_collection(
    github: bool           = True,
    hf_rss: bool           = True,
    hf_models: bool        = True,
    hf_spaces: bool        = True,
    pwc: bool              = True,
    arxiv: bool            = True,
    kaggle: bool           = True,
    zenodo: bool           = True,
    openml: bool           = True,
    max_github: int        = 100,
    max_hf_models: int     = 500,
    max_hf_spaces: int     = 200,
    max_pwc: int           = 200,
    max_arxiv: int         = 500,
    max_kaggle: int        = 100,
    max_zenodo: int        = 200,
    max_openml: int        = 500,
    reset_checkpoint: bool = False,
) -> dict:
    """
    Collecte incrémentale — 9 sources :
    GitHub · HuggingFace · ArXiv · Papers With Code
    Kaggle · Zenodo · OpenML
    Ne touche jamais aux données existantes.
    """
    init_db()
    session    = db_session()
    checkpoint = Checkpoint()

    if reset_checkpoint:
        log.info("⚠️  Checkpoint réinitialisé — collecte complète relancée")
        checkpoint.reset()

    log.info(f"Checkpoint actuel : {checkpoint.summary()}")

    diffs: dict[str, DiffCounter] = {}

    try:
        if github:
            d = DiffCounter("github")
            GitHubCollector().collect(session, checkpoint, d)
            diffs["github"] = d

        if hf_rss:
            d = DiffCounter("hf_rss")
            HuggingFaceRSSCollector().collect(session, checkpoint, d)
            diffs["hf_rss"] = d

        if hf_models:
            d = DiffCounter("hf_models")
            HuggingFaceHubCollector().collect_models(
                session, checkpoint, d, max_per_topic=max_hf_models)
            diffs["hf_models"] = d

        if hf_spaces:
            d = DiffCounter("hf_spaces")
            HuggingFaceHubCollector().collect_spaces(
                session, checkpoint, d, max_per_topic=max_hf_spaces)
            diffs["hf_spaces"] = d

        if pwc:
            d = DiffCounter("pwc")
            PapersWithCodeCollector().collect(session, checkpoint, d, max_per_task=max_pwc)
            diffs["pwc"] = d

        if arxiv:
            d = DiffCounter("arxiv")
            ArXivCollector().collect(session, checkpoint, d, max_per_category=max_arxiv)
            diffs["arxiv"] = d

        if kaggle:
            d = DiffCounter("kaggle")
            KaggleCollector().collect(session, checkpoint, d, max_per_term=max_kaggle)
            diffs["kaggle"] = d

        if zenodo:
            d = DiffCounter("zenodo")
            ZenodoCollector().collect(session, checkpoint, d, max_per_query=max_zenodo)
            diffs["zenodo"] = d

        if openml:
            d = DiffCounter("openml")
            OpenMLCollector().collect(session, checkpoint, d, max_per_tag=max_openml)
            diffs["openml"] = d

    except KeyboardInterrupt:
        log.warning("⚡ Interruption — données sauvegardées, checkpoint conservé")
        session.commit()

    finally:
        session.close()

    log.info("═" * 65)
    log.info("RAPPORT DE COLLECTE")
    log.info("═" * 65)
    total_new = total_updated = 0
    for name, d in diffs.items():
        log.info(d.report())
        total_new     += d.new
        total_updated += d.updated
    log.info("─" * 65)
    log.info(f"TOTAL  ✅ nouveaux={total_new}  🔄 mis à jour={total_updated}")
    log.info("═" * 65)

    return {
        name: {
            "new": d.new, "updated": d.updated,
            "skipped": d.skipped, "rejected": d.rejected,
        }
        for name, d in diffs.items()
    }


if __name__ == "__main__":
    run_collection()