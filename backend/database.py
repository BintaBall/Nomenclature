"""
backend/database.py — Dataset 2 via MongoDB (motor async)
==========================================================
Collection : submissions
  - description, tags, technologies
  - predicted_domain, domain_confidence
  - originality_score, originality_label
  - similar_projects (top 5 résumés)
  - is_public (contribué au dataset public)
  - submitted_at, version

Collection : contributions
  - projets approuvés pour le dataset public
  - même schéma + validated_at
"""

import os
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "aiscope")

# ── Client singleton ──────────────────────────────────────────────────────
_client: Optional[AsyncIOMotorClient] = None

def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client

def get_db():
    return get_client()[MONGO_DB]

def get_submissions():
    return get_db()["submissions"]

def get_contributions():
    return get_db()["contributions"]


# ── Helpers CRUD ──────────────────────────────────────────────────────────

async def insert_submission(data: dict) -> str:
    """Insère une soumission dans Dataset 2. Retourne l'id MongoDB."""
    doc = {
        "description":        data.get("description", ""),
        "tags":               data.get("tags", ""),
        "technologies":       data.get("technologies", ""),
        "predicted_domain":   data.get("predicted_domain", ""),
        "domain_confidence":  data.get("domain_confidence", 0.0),
        "originality_score":  data.get("originality_score", 0.0),
        "originality_label":  data.get("originality_label", ""),
        # Top 5 projets similaires résumés (pas toute la réponse)
        "similar_projects": [
            {
                "name":   p.get("name"),
                "domain": p.get("domain"),
                "score":  p.get("score"),
                "url":    p.get("url"),
            }
            for p in data.get("similar_projects", [])[:5]
        ],
        "is_public":    False,   # n'est dans le dataset public qu'après contribution
        "submitted_at": datetime.now(timezone.utc),
        "version":      1,
    }
    result = await get_submissions().insert_one(doc)
    return str(result.inserted_id)


async def get_user_submissions(limit: int = 50) -> list[dict]:
    """Récupère l'historique des soumissions (les plus récentes d'abord)."""
    cursor = get_submissions().find(
        {}, {"_id": 1, "description": 1, "predicted_domain": 1,
             "originality_score": 1, "originality_label": 1,
             "is_public": 1, "submitted_at": 1}
    ).sort("submitted_at", -1).limit(limit)

    results = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        results.append(doc)
    return results


async def contribute_submission(submission_data: dict) -> str:
    """
    Marque une soumission comme publique ET l'ajoute à la collection contributions.

    Différence claire :
      - insert_submission  → Dataset 2, privé, visible dans "Mes soumissions"
      - contribute_submission → Dataset 2 (is_public=True) + contributions (public)

    La contribution est ce qui apparaît dans la Library sous "Projets soumis".
    """
    # 1 — Mettre is_public=True sur la soumission existante si submission_id fourni
    sub_id = submission_data.get("submission_id")
    if sub_id:
        from bson import ObjectId
        await get_submissions().update_one(
            {"_id": ObjectId(sub_id)},
            {"$set": {"is_public": True, "published_at": datetime.now(timezone.utc)}}
        )

    # 2 — Créer un document dans contributions (Dataset public étendu)
    contribution = {
        "description":       submission_data.get("description", ""),
        "tags":              submission_data.get("tags", ""),
        "technologies":      submission_data.get("technologies", ""),
        "predicted_domain":  submission_data.get("predicted_domain", ""),
        "originality_score": submission_data.get("originality_score", 0.0),
        "originality_label": submission_data.get("originality_label", ""),
        "author":            submission_data.get("user_name", "Anonyme"),
        "source":            "user_submission",
        "is_public":         True,
        "validated_at":      datetime.now(timezone.utc),
    }
    result = await get_contributions().insert_one(contribution)
    return str(result.inserted_id)


async def ping_db() -> bool:
    """Vérifie la connexion MongoDB."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False


async def create_indexes():
    """Crée les index MongoDB au démarrage (idempotent)."""
    await get_submissions().create_index("submitted_at")
    await get_submissions().create_index("predicted_domain")
    await get_submissions().create_index("originality_score")
    await get_contributions().create_index("validated_at")
    await get_contributions().create_index("predicted_domain")


async def get_contributions_paged(
    domain: str = None,
    q:      str = None,
    page:   int = 1,
    limit:  int = 24,
) -> dict:
    """
    Retourne les contributions publiques (Dataset 2 visible dans Library).
    Différent de get_user_submissions qui retourne l'historique privé.
    """
    filt = {"is_public": True}
    if domain:
        filt["predicted_domain"] = domain
    if q:
        filt["$or"] = [
            {"description": {"$regex": q, "$options": "i"}},
        ]

    total  = await get_contributions().count_documents(filt)
    skip   = (page - 1) * limit
    cursor = get_contributions().find(filt).sort("validated_at", -1).skip(skip).limit(limit)

    projects = []
    async for doc in cursor:
        projects.append({
            "id":              str(doc.pop("_id")),
            "name":            doc.get("description", "")[:60] + "…",
            "description":     doc.get("description", "")[:180],
            "author":          doc.get("author", "Anonyme"),
            "url":             "",
            "domain":          doc.get("predicted_domain", ""),
            "source":          "user_submission",
            "stars":           0,
            "has_paper":       False,
            "is_deployed":     False,
            "technologies":    doc.get("technologies", ""),
            "popularity_score":doc.get("originality_score", 0),
            "originality_label": doc.get("originality_label", ""),
            "published_at":    str(doc.get("validated_at", ""))[:10],
        })

    return {"projects": projects, "total": total, "page": page, "limit": limit}