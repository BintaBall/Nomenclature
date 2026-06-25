"""
backend/auth.py — Authentification JWT + GitHub OAuth
======================================================
Deux méthodes d'authentification :
  1. Email/password  → POST /auth/register, POST /auth/login
  2. GitHub OAuth    → GET /auth/github, GET /auth/github/callback

Variables .env requises pour GitHub OAuth :
  GITHUB_CLIENT_ID     → depuis github.com/settings/developers
  GITHUB_CLIENT_SECRET → depuis github.com/settings/developers
  FRONTEND_URL         → http://localhost:5173 (redirect après login)
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import httpx
from fastapi import HTTPException, Depends
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

from database import get_db

SECRET_KEY    = os.getenv("JWT_SECRET", "aiscope-dev-secret-change-in-prod-2024")
ALGORITHM     = "HS256"
TOKEN_EXPIRY  = int(os.getenv("JWT_EXPIRY_HOURS", "72"))
FRONTEND_URL  = os.getenv("FRONTEND_URL", "https://nomenclature.glybette.com")

GH_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID", "")
GH_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GH_REDIRECT_URI  = os.getenv("GITHUB_REDIRECT_URI", "http://51.210.178.46:8000/auth/github/callback")

security = HTTPBearer(auto_error=False)


# ── Schémas ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    str
    password: str
    name:     str = ""

class LoginRequest(BaseModel):
    email:    str
    password: str


# ── JWT helpers ───────────────────────────────────────────────────────────

def hash_password(p: str) -> str:
    """Hash un mot de passe avec bcrypt (max 72 caractères)"""
    # Bcrypt ne supporte que 72 caractères
    password_bytes = p[:72].encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')

def verify_password(plain: str, hashed: str) -> bool:
    """Vérifie un mot de passe avec bcrypt"""
    password_bytes = plain[:72].encode('utf-8')
    hashed_bytes = hashed.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)

def create_token(user_id: str, email: str, name: str = "") -> str:
    expire  = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY)
    payload = {"sub": user_id, "email": email, "name": name, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(401, "Token invalide ou expiré")


# ── Dépendances FastAPI ───────────────────────────────────────────────────

async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    if not creds:
        return None
    try:
        payload = decode_token(creds.credentials)
        return {
            "id":    payload["sub"],
            "email": payload.get("email", ""),
            "name":  payload.get("name", ""),
        }
    except Exception:
        return None


# ── Email/password ────────────────────────────────────────────────────────

async def register_user(email: str, password: str, name: str) -> dict:
    users = get_db()["users"]

    if len(password) < 6:
        raise HTTPException(400, "Mot de passe trop court (minimum 6 caractères)")
    if len(password) > 72:
        raise HTTPException(400, "Mot de passe trop long (maximum 72 caractères)")

    existing = await users.find_one({"email": email.lower()})
    if existing:
        raise HTTPException(400, "Cet email est déjà utilisé")

    doc = {
        "email":         email.lower(),
        "password_hash": hash_password(password),
        "name":          name.strip() or email.split("@")[0],
        "provider":      "email",
        "created_at":    datetime.now(timezone.utc),
    }
    result  = await users.insert_one(doc)
    user_id = str(result.inserted_id)

    # Index unique sur email (idempotent)
    await users.create_index("email", unique=True)

    token = create_token(user_id, doc["email"], doc["name"])
    return {
        "token": token,
        "user":  {"id": user_id, "email": doc["email"], "name": doc["name"]},
    }


async def login_user(email: str, password: str) -> dict:
    users = get_db()["users"]
    user  = await users.find_one({"email": email.lower()})

    if not user:
        raise HTTPException(401, "Email ou mot de passe incorrect")

    # Compte GitHub sans mot de passe
    if not user.get("password_hash"):
        raise HTTPException(400, "Ce compte utilise la connexion GitHub. Utilisez 'Continuer avec GitHub'.")

    if not verify_password(password, user["password_hash"]):
        raise HTTPException(401, "Email ou mot de passe incorrect")

    user_id = str(user["_id"])
    token   = create_token(user_id, user["email"], user.get("name", ""))
    return {
        "token": token,
        "user":  {"id": user_id, "email": user["email"], "name": user.get("name", "")},
    }


# ── GitHub OAuth ──────────────────────────────────────────────────────────

def github_login_url() -> str:
    """Retourne l'URL d'autorisation GitHub."""
    if not GH_CLIENT_ID:
        raise HTTPException(500, "GitHub OAuth non configuré (GITHUB_CLIENT_ID manquant dans .env)")
    return (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GH_CLIENT_ID}"
        f"&redirect_uri={GH_REDIRECT_URI}"
        f"&scope=user:email"
    )


async def github_callback(code: str) -> dict:
    """
    Échange le code GitHub contre un token d'accès,
    récupère le profil utilisateur, crée ou met à jour le compte.
    """
    if not GH_CLIENT_ID or not GH_CLIENT_SECRET:
        raise HTTPException(500, "GitHub OAuth non configuré")

    async with httpx.AsyncClient() as client:
        # 1 — Échange code → access_token
        token_res = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id":     GH_CLIENT_ID,
                "client_secret": GH_CLIENT_SECRET,
                "code":          code,
                "redirect_uri":  GH_REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        token_data   = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(400, f"GitHub OAuth échoué : {token_data.get('error_description', 'code invalide')}")

        gh_headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

        # 2 — Récupérer le profil
        profile_res = await client.get("https://api.github.com/user", headers=gh_headers, timeout=10)
        profile     = profile_res.json()

        # 3 — Récupérer l'email (peut être privé)
        email = profile.get("email")
        if not email:
            email_res = await client.get("https://api.github.com/user/emails", headers=gh_headers, timeout=10)
            emails    = email_res.json()
            primary   = next((e for e in emails if e.get("primary") and e.get("verified")), None)
            email     = primary["email"] if primary else f"gh_{profile['id']}@github.noreply"

    # 4 — Upsert en base
    users    = get_db()["users"]
    gh_id    = str(profile["id"])
    existing = await users.find_one({"$or": [{"github_id": gh_id}, {"email": email.lower()}]})

    name     = profile.get("name") or profile.get("login", "")
    avatar   = profile.get("avatar_url", "")

    if existing:
        user_id = str(existing["_id"])
        await users.update_one(
            {"_id": existing["_id"]},
            {"$set": {
                "github_id":  gh_id,
                "name":       name,
                "avatar_url": avatar,
                "last_login": datetime.now(timezone.utc),
            }}
        )
    else:
        doc = {
            "email":      email.lower(),
            "github_id":  gh_id,
            "name":       name,
            "avatar_url": avatar,
            "provider":   "github",
            "created_at": datetime.now(timezone.utc),
        }
        result  = await users.insert_one(doc)
        user_id = str(result.inserted_id)

    token = create_token(user_id, email, name)
    return {
        "token": token,
        "user":  {"id": user_id, "email": email, "name": name, "avatar": avatar},
    }