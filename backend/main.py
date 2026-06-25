"""
backend/main.py — AIScope API
==============================
Endpoints :
  GET  /health
  GET  /stats
  POST /search          → similarité interne
  POST /external        → recherche externe manuelle
  GET  /library         → dataset public paginé
  GET  /insights        → agrégations MongoDB
  POST /history/save    → sauvegarder analyse (auth optionnelle)
  GET  /history/mine    → historique utilisateur (auth requise)
  DELETE /history/{id}  → supprimer soumission (auth requise)
  GET  /history/{id}    → détail soumission (auth requise)
  POST /contribute      → contribuer au dataset public
  POST /auth/register   → créer compte email
  POST /auth/login      → connexion email/password
  GET  /auth/me         → profil utilisateur
  GET  /auth/github     → initier OAuth GitHub
  GET  /auth/github/callback → retour OAuth GitHub
"""

import logging
import os
import time
import urllib.parse
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from auth import (
    RegisterRequest, LoginRequest,
    register_user, login_user,
    get_current_user,
    github_login_url, github_callback,
)

load_dotenv()
log = logging.getLogger("aiscope")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

CSV_PATH      = os.getenv("CSV_PATH",      "../data/ai_projects_cleaned.csv")
BUNDLE_PATH   = os.getenv("BUNDLE_PATH",   "../data/xgboost_bundle.pkl")
TRIGGER_SCORE = float(os.getenv("TRIGGER_SCORE", "0.45"))
FRONTEND_URL  = os.getenv("FRONTEND_URL",  "https://nomenclature.glybette.com")

_sim_engine  = None
_ext_engine  = None
_df_cache: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sim_engine, _ext_engine
    log.info("Démarrage AIScope backend…")
    t0 = time.time()

    from similarity_light import SimilarityEngineLight
    from external_search  import ExternalSearchEngine

    _sim_engine = SimilarityEngineLight(top_k=10)
    _sim_engine.load(CSV_PATH, BUNDLE_PATH)
    _ext_engine = ExternalSearchEngine(trigger_score=TRIGGER_SCORE)

    from database import create_indexes, ping_db
    mongo_ok = await ping_db()
    if mongo_ok:
        await create_indexes()
        log.info("MongoDB connecté ✅")
    else:
        log.warning("MongoDB inaccessible — historique désactivé")

    log.info(f"Prêt en {time.time()-t0:.1f}s")
    yield
    log.info("Arrêt propre.")


app = FastAPI(
    title="AIScope API",
    description="Similarité IA + Dataset 2 MongoDB",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "https://nomenclature.glybette.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas Pydantic ──────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    description:  str  = Field(..., min_length=20, max_length=2000)
    tags:         str  = Field("", max_length=500)
    technologies: str  = Field("", max_length=500)
    force_domain: Optional[str] = None
    top_k:        int  = Field(10, ge=1, le=25)

class ExternalRequest(BaseModel):
    description:  str
    tags:         str = ""
    technologies: str = ""
    domain:       str = ""

class HistoryRequest(BaseModel):
    description:       str
    tags:              str   = ""
    technologies:      str   = ""
    predicted_domain:  str   = ""
    domain_confidence: float = 0.0
    originality_score: float = 0.0
    originality_label: str   = ""
    similar_projects:  list  = []

class ContributeRequest(BaseModel):
    description:       str
    tags:              str   = ""
    technologies:      str   = ""
    predicted_domain:  str   = ""
    originality_score: float = 0.0
    originality_label: str   = ""
    submission_id:     Optional[str] = None


# ── Health + Stats ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from database import ping_db
    mongo_ok = await ping_db()
    if not _sim_engine:
        raise HTTPException(503, "Moteur non initialisé")
    s = _sim_engine.stats()
    return {
        "status":   "ok",
        "projects": s.get("total_projects", 0),
        "mongodb":  mongo_ok,
        "model":    s.get("uses_faiss", False),
    }


@app.get("/stats")
async def stats():
    if not _sim_engine:
        raise HTTPException(503, "Moteur non initialisé")
    return _sim_engine.stats()


# ── Recherche similarité interne ──────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest):
    if not _sim_engine:
        raise HTTPException(503, "Moteur non initialisé")

    t0 = time.perf_counter()
    internal = _sim_engine.search(
        query=req.description,
        tags=req.tags,
        technologies=req.technologies,
        force_domain=req.force_domain,
    )

    # Recherche externe désactivée en auto — déclenchée manuellement
    external_triggered = False
    external_results   = []

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "query":              req.description[:100],
        "predicted_domain":   internal.predicted_domain,
        "domain_confidence":  round(internal.domain_confidence, 4),
        "originality_score":  internal.originality_score,
        "originality_label":  internal.originality_label,
        "dual_search":        internal.dual_search,
        "search_domains":     internal.search_domains,
        "similar_projects": [
            {
                "rank":             p.rank,
                "name":             p.name,
                "full_name":        p.full_name,
                "description":      p.description,
                "domain":           p.domain,
                "author":           p.author,
                "url":              p.url,
                "stars":            p.stars,
                "has_paper":        p.has_paper,
                "is_deployed":      p.is_deployed,
                "technologies":     p.technologies,
                "score":            round(p.score, 4),
                "similarity_label": p.similarity_label,
                "same_domain":      p.same_domain,
            }
            for p in internal.similar_projects
        ],
        "related_domains":    internal.related_domains,
        "external_triggered": external_triggered,
        "external_results":   external_results,
        "suggestions":        internal.suggestions,
        "processing_time_ms": elapsed,
    }


# ── Recherche externe manuelle ────────────────────────────────────────────

@app.post("/external")
async def external_search(req: ExternalRequest):
    if not _ext_engine:
        raise HTTPException(503, "Moteur externe non initialisé")

    try:
        ext = _ext_engine.search(
            query=req.description,
            domain=req.domain,
            tags=req.tags,
            technologies=req.technologies,
            triggered_by="manual",
        )
        return {
            "results": [
                {
                    "title":       r.title,
                    "description": r.description[:150],
                    "url":         r.url,
                    "source":      r.source,
                    "author":      r.author,
                    "stars":       r.stars,
                    "has_paper":   r.has_paper,
                    "relevance":   r.relevance,
                }
                for r in ext.results
            ],
            "sources_used": ext.sources_used,
            "total_found":  ext.total_found,
        }
    except Exception as e:
        log.warning(f"Recherche externe échouée : {e}")
        return {"results": [], "sources_used": [], "total_found": 0}


# ── Library (Dataset 1 — CSV) ─────────────────────────────────────────────

@app.get("/library")
async def library(
    dataset: str           = Query("public"),
    domain:  Optional[str] = Query(None),
    source:  Optional[str] = Query(None),
    sort:    str           = Query("popularity"),
    q:       Optional[str] = Query(None),
    page:    int           = Query(1,  ge=1),
    limit:   int           = Query(24, ge=1, le=100),
):
    # Onglet contributions → MongoDB
    if dataset == "contributed":
        try:
            from database import get_contributions_paged
            return await get_contributions_paged(domain=domain, q=q, page=page, limit=limit)
        except Exception as e:
            log.warning(f"Contributions MongoDB error: {e}")
            return {"projects": [], "total": 0, "page": page, "limit": limit}

    # Dataset public → CSV
    global _df_cache
    if not _sim_engine or _sim_engine.df is None:
        raise HTTPException(503, "Dataset non chargé")

    if _df_cache is None:
        _df_cache = _sim_engine.df.copy()

    df = _df_cache
    if domain:
        df = df[df["domain"] == domain]
    if source:
        df = df[df["source"] == source]
    if q:
        mask = (
            df["description"].str.contains(q, case=False, na=False) |
            df["name"].str.contains(q, case=False, na=False)
        )
        df = df[mask]

    sort_map = {"popularity": "popularity_score", "stars": "stars", "recent": "published_at"}
    sort_col = sort_map.get(sort, "popularity_score")
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    total   = len(df)
    start   = (page - 1) * limit
    page_df = df.iloc[start: start + limit]

    projects = []
    for _, row in page_df.iterrows():
        projects.append({
            "id":               str(row.get("id", "")),
            "name":             str(row.get("name", "")),
            "description":      str(row.get("description", ""))[:180],
            "author":           str(row.get("author", "")),
            "url":              str(row.get("url", "")),
            "domain":           str(row.get("domain", "")),
            "source":           str(row.get("source", "")),
            "stars":            int(row.get("stars", 0)),
            "has_paper":        bool(row.get("has_paper", False)),
            "is_deployed":      bool(row.get("is_deployed", False)),
            "technologies":     str(row.get("technologies", "")),
            "popularity_score": float(row.get("popularity_score", 0)),
            "published_at":     str(row.get("published_at", ""))[:10],
        })

    return {"projects": projects, "total": total, "page": page, "limit": limit}


# ── Historique utilisateur (Dataset 2 — MongoDB) ──────────────────────────

@app.post("/history/save")
async def save_history(req: HistoryRequest, user=Depends(get_current_user)):
    """Sauvegarder une analyse dans l'historique de l'utilisateur connecté."""
    try:
        from database import insert_submission
        data = req.model_dump()
        # ✅ Utiliser l'ID de l'utilisateur connecté
        data["user_id"] = user["id"] if user else "anonymous"
        doc_id = await insert_submission(data)
        return {"id": doc_id, "message": "Sauvegardé"}
    except Exception as e:
        log.warning(f"MongoDB save error: {e}")
        return {"id": None, "message": "Non sauvegardé"}


@app.get("/history/mine")
async def get_my_history(user=Depends(get_current_user)):
    """Historique personnel de l'utilisateur connecté."""
    try:
        from database import get_user_submissions
        # ✅ Utiliser l'ID de l'utilisateur connecté
        if not user:
            return {"submissions": []}
        subs = await get_user_submissions(user_id=user["id"], limit=100)
        return {"submissions": subs}
    except Exception as e:
        log.warning(f"MongoDB history error: {e}")
        return {"submissions": []}


@app.delete("/history/{submission_id}")
async def delete_submission_ep(submission_id: str, user=Depends(get_current_user)):
    """Supprimer une soumission (seulement si elle appartient à l'utilisateur)."""
    try:
        from database import delete_submission
        if not user:
            raise HTTPException(401, "Non connecté")
        deleted = await delete_submission(submission_id, user["id"])
        if not deleted:
            raise HTTPException(404, "Soumission introuvable")
        return {"message": "Soumission supprimée"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/history/{submission_id}")
async def get_submission_detail(submission_id: str, user=Depends(get_current_user)):
    """Détail d'une soumission (seulement si elle appartient à l'utilisateur)."""
    try:
        from database import get_submission_by_id
        if not user:
            raise HTTPException(401, "Non connecté")
        doc = await get_submission_by_id(submission_id, user["id"])
        if not doc:
            raise HTTPException(404, "Soumission introuvable")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Contribution au dataset public ────────────────────────────────────────

@app.post("/contribute")
async def contribute(req: ContributeRequest):
    if req.originality_score < 0.45:
        raise HTTPException(
            400,
            f"Score insuffisant ({req.originality_score:.0%} < 45%). "
            "Seuls les projets suffisamment originaux peuvent enrichir le dataset."
        )
    try:
        from database import contribute_submission
        contrib_id = await contribute_submission(req.model_dump())
        return {
            "id":      contrib_id,
            "message": "Merci ! Votre projet est maintenant visible dans les contributions.",
        }
    except Exception as e:
        log.error(f"Erreur contribution MongoDB: {e}")
        raise HTTPException(500, "Erreur lors de la contribution")


# ── Insights (agrégations MongoDB) ───────────────────────────────────────

@app.get("/insights")
async def insights():
    try:
        from database import get_submissions, get_contributions

        subs  = get_submissions()
        conts = get_contributions()

        total_submissions   = await subs.count_documents({})
        total_contributions = await conts.count_documents({})

        pipeline_avg = [{"$group": {"_id": None, "avg": {"$avg": "$originality_score"}}}]
        avg_res   = await subs.aggregate(pipeline_avg).to_list(1)
        avg_score = round(avg_res[0]["avg"], 3) if avg_res else None

        pipeline_dist = [{"$group": {"_id": "$originality_label", "count": {"$sum": 1}}}]
        dist_res  = await subs.aggregate(pipeline_dist).to_list(10)
        orig_dist = {d["_id"]: d["count"] for d in dist_res if d["_id"]}

        pipeline_domains = [
            {"$group": {"_id": "$predicted_domain", "count": {"$sum": 1}}},
            {"$sort":  {"count": -1}},
            {"$limit": 8},
        ]
        top_domains = await subs.aggregate(pipeline_domains).to_list(8)

        return {
            "total_submissions":        total_submissions,
            "total_contributions":      total_contributions,
            "avg_originality_score":    avg_score,
            "originality_distribution": orig_dist,
            "top_searched_domains":     top_domains,
        }
    except Exception as e:
        log.warning(f"Insights MongoDB error: {e}")
        return {
            "total_submissions": 0, "total_contributions": 0,
            "avg_originality_score": None,
            "originality_distribution": {}, "top_searched_domains": [],
        }


# ── Auth endpoints ────────────────────────────────────────────────────────

from auth import (
    RegisterRequest, LoginRequest,
    register_user, login_user,
    get_current_user,
    github_login_url, github_callback,
)


@app.post("/auth/register", status_code=201)
async def register(req: RegisterRequest):
    """Crée un compte email/password — retourne token + user."""
    return await register_user(req.email, req.password, req.name)


@app.post("/auth/login")
async def login(req: LoginRequest):
    """Connexion email/password — retourne token + user."""
    return await login_user(req.email, req.password)


@app.get("/auth/me")
async def me(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Non connecté")
    return user


@app.get("/auth/github")
async def github_login():
    """Redirige vers GitHub pour autorisation OAuth."""
    url = github_login_url()
    return RedirectResponse(url)


@app.get("/auth/github/callback")
async def github_cb(code: str = None, error: str = None):
    """
    Callback GitHub — échange le code, crée/met à jour l'utilisateur,
    redirige vers /login/callback?token=...
    """
    frontend = os.getenv("FRONTEND_URL", "https://nomenclature.glybette.com")
    if error or not code:
        return RedirectResponse(f"{frontend}/login?error=github_denied")
    try:
        result = await github_callback(code)
        u  = result["user"]
        qs = urllib.parse.urlencode({
            "token":  result["token"],
            "name":   u.get("name", ""),
            "email":  u.get("email", ""),
            "avatar": u.get("avatar", ""),
        })
        return RedirectResponse(f"{frontend}/login/callback?{qs}")
    except Exception as e:
        log.error(f"GitHub OAuth error: {e}")
        return RedirectResponse(f"{frontend}/login?error=github_failed")