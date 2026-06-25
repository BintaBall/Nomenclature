"""
models.py — Schéma SQLAlchemy complet pour les projets IA collectés.
Supporte SQLite (dev) et PostgreSQL (prod) via DATABASE_URL.
"""

import os
from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    Boolean, Text, DateTime, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_projects.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def db_session():
    return SessionLocal()


def init_db():
    Base.metadata.create_all(bind=engine)


class AIProject(Base):
    """
    Représentation normalisée d'un projet IA.
    Toutes les sources (GitHub, HuggingFace, RSS) alimentent cette table.
    """
    __tablename__ = "ai_projects"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_fingerprint"),
        Index("ix_source", "source"),
        Index("ix_popularity", "popularity_score"),
        Index("ix_published", "published_at"),
        Index("ix_is_deployed", "is_deployed"),
    )

    # ── Identité ────────────────────────────────────────────────────────────
    id = Column(Integer, primary_key=True, autoincrement=True)
    fingerprint    = Column(String(64), unique=True, nullable=False, comment="SHA256(source::source_id)")
    source         = Column(String(32), nullable=False, comment="github | hf_model | hf_space | hf_rss")
    source_id      = Column(String(256), nullable=False, comment="ID natif dans la source")

    # ── Métadonnées de base ──────────────────────────────────────────────────
    name           = Column(String(256), nullable=False)
    full_name      = Column(String(512), nullable=False, comment="owner/repo ou hf_model_id")
    description    = Column(Text, nullable=False, comment="OBLIGATOIRE — filtrée à l'insertion")
    readme_excerpt = Column(Text, nullable=True)
    author         = Column(String(256), nullable=True)
    url            = Column(String(1024), nullable=True)

    # ── Taxonomie ────────────────────────────────────────────────────────────
    tags           = Column(Text, nullable=True, comment="JSON list of tags")
    technologies   = Column(Text, nullable=True, comment="JSON list: pytorch, tensorflow, transformers…")
    language       = Column(String(64), nullable=True, comment="Langage principal (Python, R…)")
    pipeline_tag   = Column(String(128), nullable=True, comment="HuggingFace pipeline tag")
    domain         = Column(String(128), nullable=True, comment="NLP / CV / Audio / RL / Multimodal…")

    # ── Popularité ───────────────────────────────────────────────────────────
    stars          = Column(Integer, default=0)
    forks          = Column(Integer, default=0)
    watchers       = Column(Integer, default=0)
    open_issues    = Column(Integer, default=0)
    downloads      = Column(Integer, default=0, comment="HuggingFace downloads")
    popularity_score = Column(Float, default=0.0, comment="Score normalisé [0-1]")

    # ── Statut / déploiement ─────────────────────────────────────────────────
    is_deployed    = Column(Boolean, default=False, comment="Projet actif / en production")
    has_demo       = Column(Boolean, default=False, comment="Démo publique disponible")
    demo_url       = Column(String(1024), nullable=True)
    is_fork        = Column(Boolean, default=False)
    is_archived    = Column(Boolean, default=False)
    has_paper      = Column(Boolean, default=False, comment="Lié à un paper ArXiv / PDF")
    paper_url      = Column(String(1024), nullable=True)
    has_dataset    = Column(Boolean, default=False)
    dataset_url    = Column(String(1024), nullable=True)

    # ── Qualité du code ──────────────────────────────────────────────────────
    license        = Column(String(64), nullable=True)
    has_tests      = Column(Boolean, default=False)
    has_ci         = Column(Boolean, default=False)
    has_docker     = Column(Boolean, default=False)
    has_requirements = Column(Boolean, default=False)

    # ── Temporalité ──────────────────────────────────────────────────────────
    published_at   = Column(String(32), nullable=True)
    updated_at     = Column(String(32), nullable=True)
    collected_at   = Column(String(32), nullable=False)

    # ── Embeddings (Étape 2) ─────────────────────────────────────────────────
    embedding      = Column(Text, nullable=True, comment="JSON float array — à remplir en Étape 2")

    def to_dict(self) -> dict:
        import json as _json
        return {
            "id": self.id,
            "source": self.source,
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "author": self.author,
            "url": self.url,
            "tags": _json.loads(self.tags) if self.tags else [],
            "technologies": _json.loads(self.technologies) if self.technologies else [],
            "language": self.language,
            "pipeline_tag": self.pipeline_tag,
            "domain": self.domain,
            "stars": self.stars,
            "forks": self.forks,
            "downloads": self.downloads,
            "popularity_score": round(self.popularity_score or 0, 4),
            "is_deployed": self.is_deployed,
            "has_demo": self.has_demo,
            "demo_url": self.demo_url,
            "is_fork": self.is_fork,
            "is_archived": self.is_archived,
            "has_paper": self.has_paper,
            "license": self.license,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "collected_at": self.collected_at,
        }

    def __repr__(self):
        return f"<AIProject [{self.source}] {self.full_name} ★{self.stars}>"