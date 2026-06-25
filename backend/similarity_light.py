"""
similarity_light.py — Moteur de similarité léger (sans sentence-transformers)
===============================================================================
Utilise le TF-IDF + XGBoost déjà fitté dans le notebook de classification.
Démarrage <1s, RAM ~80MB, recherche ~5ms.

Dépendances : pandas scikit-learn faiss-cpu numpy
    pip install faiss-cpu scikit-learn pandas numpy

Usage :
    from similarity_light import SimilarityEngineLight
    engine = SimilarityEngineLight()
    engine.load("ai_projects.csv", bundle_path="xgboost_bundle.pkl")
    result = engine.search("détection cancer pulmonaire CNN attention")
    print(result)

CLI :
    python similarity_light.py "brain tumor detection MRI" --csv ai_projects.csv --bundle xgboost_bundle.pkl
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

log = logging.getLogger("similarity_light")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
)

# ── Seuils calibrés par domaine (issus de l'analyse UMAP/silhouette) ──────────

DOMAIN_THRESHOLDS: dict[str, dict] = {
    "Multimodal":           {"very_similar": 0.82, "similar": 0.60, "original": 0.45},
    "Audio / Speech":       {"very_similar": 0.80, "similar": 0.58, "original": 0.43},
    "Graph / Network":      {"very_similar": 0.78, "similar": 0.55, "original": 0.40},
    "3D / Point Cloud":     {"very_similar": 0.76, "similar": 0.52, "original": 0.38},
    "Generative AI":        {"very_similar": 0.75, "similar": 0.52, "original": 0.38},
    "Medical / Healthcare": {"very_similar": 0.75, "similar": 0.52, "original": 0.38},
    "Finance":              {"very_similar": 0.72, "similar": 0.48, "original": 0.35},
    "NLP / Text":           {"very_similar": 0.72, "similar": 0.48, "original": 0.35},
    "Computer Vision":      {"very_similar": 0.70, "similar": 0.46, "original": 0.33},
    "Robotics / RL":        {"very_similar": 0.68, "similar": 0.44, "original": 0.30},
    "Time Series":          {"very_similar": 0.72, "similar": 0.48, "original": 0.35},
    "Other":                {"very_similar": 0.68, "similar": 0.44, "original": 0.30},
}
DEFAULT_THRESHOLDS = {"very_similar": 0.75, "similar": 0.50, "original": 0.35}

OVERLAPPING_PAIRS: list[frozenset] = [
    frozenset({"Computer Vision",     "Robotics / RL"}),
    frozenset({"Medical / Healthcare","3D / Point Cloud"}),
    frozenset({"Medical / Healthcare","Other"}),
    frozenset({"Graph / Network",     "Other"}),
    frozenset({"NLP / Text",          "Finance"}),
    frozenset({"NLP / Text",          "Multimodal"}),
    frozenset({"Generative AI",       "Computer Vision"}),
]
DUAL_SEARCH_CONFIDENCE = 0.70

_HF_NOISE = re.compile(
    r'^(region:|license:|arxiv:|doi:|base_model:|dataset:|language:|'
    r'endpoints_compatible|safetensors|gguf|autotrain_compatible|'
    r'has_space|generated_from_trainer|not-for-all-audiences|'
    r'trl|peft|unsloth|merge|adapter)'
)

def _parse_list(s, sep=";", clean=False):
    if not isinstance(s, str) or not s.strip():
        return []
    items = [t.strip() for t in s.split(sep) if t.strip()]
    if clean:
        items = [t for t in items if not _HF_NOISE.match(t.lower())]
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimilarProject:
    rank:             int
    name:             str
    full_name:        str
    description:      str
    domain:           str
    author:           str
    url:              str
    stars:            int
    has_paper:        bool
    is_deployed:      bool
    technologies:     list
    score:            float
    similarity_label: str
    same_domain:      bool


@dataclass
class SimilarityResult:
    query:             str
    predicted_domain:  str
    domain_confidence: float
    similar_projects:  list
    related_domains:   list
    originality_score: float
    originality_label: str
    suggestions:       list
    dual_search:       bool
    search_domains:    list   # domaines effectivement cherchés

    def __str__(self):
        lines = [
            f"\nQuery            : {self.query[:80]}",
            f"Domaine prédit   : {self.predicted_domain} "
            f"({self.domain_confidence:.0%} confiance)",
            f"Originalité      : {self.originality_score:.3f} — {self.originality_label}",
            f"Dual search      : {'oui' if self.dual_search else 'non'}",
            "",
            f"{'Rk':>3}  {'Score':>6}  {'Dom':>5}  Nom",
            "─" * 65,
        ]
        for p in self.similar_projects:
            same = "✅" if p.same_domain else "↗ "
            lines.append(
                f"{p.rank:>3}  {p.score:.4f}  {same}  "
                f"{p.name[:35]:<35}  ★{p.stars}"
            )
        lines += ["", *self.suggestions]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Moteur léger
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityEngineLight:
    """
    Moteur de similarité basé sur TF-IDF + cosine (sklearn + faiss).
    Réutilise le TF-IDF du bundle XGBoost — aucun modèle lourd requis.
    """

    def __init__(self, top_k: int = 10):
        self.top_k    = top_k
        self.df: Optional[pd.DataFrame] = None
        self.index    = None       # faiss index ou matrice numpy
        self.X_norm   = None       # matrice normalisée (fallback sans faiss)
        self.xgb_model  = None
        self.tfidf      = None
        self.scaler     = None
        self.classes: list[str] = []
        self._top_techs: list[str] = []
        self._use_faiss = False

        # Vérif faiss
        try:
            import faiss as _faiss
            self._faiss  = _faiss
            self._use_faiss = True
        except ImportError:
            log.warning("faiss absent — fallback numpy (plus lent). pip install faiss-cpu")
            self._faiss = None

    # ── Chargement ────────────────────────────────────────────────────────

    def load(self, csv_path: str, bundle_path: str):
        """
        Charge le dataset CSV + le bundle XGBoost.

        Parameters
        ----------
        csv_path    : chemin vers ai_projects.csv
        bundle_path : chemin vers xgboost_bundle.pkl
                      (produit par save_model_bundle.py Cell A)
        """
        # Dataset
        log.info(f"Chargement CSV : {csv_path}")
        self.df = pd.read_csv(csv_path)
        self._clean_df()
        log.info(f"  {len(self.df):,} projets, {self.df['domain'].nunique()} domaines")

        # Bundle
        log.info(f"Chargement bundle : {bundle_path}")
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            self.xgb_model = bundle.get("model")
            self.tfidf     = bundle.get("tfidf")
            self.scaler    = bundle.get("scaler")
            self.classes   = list(bundle.get("classes", []))
        else:
            raise ValueError(
                "Format bundle invalide. Utilise save_model_bundle.py Cell A "
                "pour créer un bundle complet (model + tfidf + scaler + classes)."
            )

        # Top technologies pour multi-hot
        all_t = [t for techs in self.df["techs_list"] for t in techs]
        self._top_techs = [t for t, _ in Counter(all_t).most_common(30)]

        # Index de similarité
        self._build_index()
        log.info("Moteur prêt.")

    def _clean_df(self):
        for col in ["is_deployed","has_demo","is_fork","is_archived","has_paper"]:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col].astype(str).str.lower()
                    .map({"true":True,"false":False,"1":True,"0":False})
                    .fillna(False)
                )
        for col in ["stars","forks","downloads","popularity_score"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        self.df["tags_clean"] = self.df["tags"].apply(
            lambda x: _parse_list(x, clean=True))
        self.df["techs_list"] = self.df["technologies"].apply(_parse_list)

        # Texte combiné pour TF-IDF
        self.df["combined_text"] = (
            self.df["description"].fillna("") + " " +
            self.df["tags"].fillna("").str.replace(";", " ") + " " +
            self.df["technologies"].fillna("").str.replace(";", " ")
        )

    def _build_index(self):
        """
        Transforme le dataset via TF-IDF et construit l'index FAISS.
        Cache la matrice normalisée dans un fichier .npy pour éviter
        de la reconstruire à chaque redémarrage.
        """
        import hashlib, os
        # Cache key basé sur la taille du CSV + nom du bundle
        cache_key  = hashlib.md5(f"{len(self.df)}_{len(self._top_techs)}".encode()).hexdigest()[:10]
        cache_path = f".faiss_cache_{cache_key}.npy"

        if os.path.exists(cache_path):
            log.info(f"Chargement index depuis cache : {cache_path}")
            self.X_norm = np.load(cache_path).astype(np.float32)
        else:
            log.info("Construction de l'index TF-IDF...")
            import time; t0 = time.time()

            X_tfidf = self.tfidf.transform(self.df["combined_text"]).toarray()

            # Multi-hot technologies — vectorisé (évite la double boucle Python lente)
            tech_idx = {t: i for i, t in enumerate(self._top_techs)}
            tech_mat = np.zeros((len(self.df), len(self._top_techs)), dtype=np.float32)
            for row_i, techs in enumerate(self.df["techs_list"]):
                for t in techs:
                    col_i = tech_idx.get(t)
                    if col_i is not None:
                        tech_mat[row_i, col_i] = 1.0

            X = np.hstack([X_tfidf, tech_mat]).astype(np.float32)
            self.X_norm = normalize(X, norm="l2")

            # Sauvegarder le cache
            np.save(cache_path, self.X_norm)
            log.info(f"Index construit en {time.time()-t0:.1f}s — cache sauvegardé : {cache_path}")

        if self._use_faiss:
            dim = self.X_norm.shape[1]
            self.index = self._faiss.IndexFlatIP(dim)
            self.index.add(self.X_norm)
            log.info(f"Index FAISS : {self.index.ntotal:,} vecteurs ({dim}D)")
        else:
            log.info(f"Index numpy : {self.X_norm.shape[0]:,} × {self.X_norm.shape[1]}D")

    # ── Prédiction domaine ────────────────────────────────────────────────

    def _predict_domain(self, text, tags="", technologies=""):
        combined  = f"{text} {tags.replace(';',' ')} {technologies.replace(';',' ')}"
        tfidf_vec = self.tfidf.transform([combined]).toarray()

        tech_idx = {t: i for i, t in enumerate(self._top_techs)}
        tech_vec = np.zeros((1, len(self._top_techs)))
        for t in _parse_list(technologies):
            col_i = tech_idx.get(t)
            if col_i is not None:
                tech_vec[0, col_i] = 1.0

        # Nombre de features structurelles attendu par le scaler/modèle
        # = total features du bundle - tfidf_features - tech_features
        n_tfidf = tfidf_vec.shape[1]
        n_tech  = tech_vec.shape[1]
        try:
            n_total   = self.xgb_model.n_features_in_
            n_struct  = max(n_total - n_tfidf - n_tech, 0)
        except AttributeError:
            n_struct = 12  # fallback

        struct = np.zeros((1, n_struct))
        X      = np.hstack([struct, tfidf_vec, tech_vec])

        # Scaler optionnel — on ignore les warnings de version
        if self.scaler is not None:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X = self.scaler.transform(X)
            except Exception:
                pass  # si le scaler plante, on continue sans

        probs   = self.xgb_model.predict_proba(X)[0]
        classes = self.classes or list(self.xgb_model.classes_)
        pmap    = dict(zip(classes, probs.tolist()))
        top_idx = int(np.argmax(probs))
        return classes[top_idx], float(probs[top_idx]), pmap

    def _get_search_domains(self, domain, confidence):
        if confidence >= DUAL_SEARCH_CONFIDENCE or domain == "Unknown":
            return [domain], False
        for pair in OVERLAPPING_PAIRS:
            if domain in pair:
                others = set(pair) - {domain}
                other  = others.pop()
                return [domain, other], True
        return [domain], False

    # ── Vecteur requête ───────────────────────────────────────────────────

    def _query_vector(self, text, tags="", technologies=""):
        combined  = f"{text} {tags.replace(';',' ')} {technologies.replace(';',' ')}"
        tfidf_vec = self.tfidf.transform([combined]).toarray()

        tech_idx = {t: i for i, t in enumerate(self._top_techs)}
        tech_vec = np.zeros((1, len(self._top_techs)))
        for t in _parse_list(technologies):
            col_i = tech_idx.get(t)
            if col_i is not None:
                tech_vec[0, col_i] = 1.0

        X = np.hstack([tfidf_vec, tech_vec]).astype(np.float32)
        return normalize(X, norm="l2")

    # ── Recherche ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        tags: str         = "",
        technologies: str = "",
        force_domain: str = None,
    ) -> SimilarityResult:

        # 1 — Domaine
        if force_domain:
            domain, conf, pmap = force_domain, 1.0, {force_domain: 1.0}
        else:
            domain, conf, pmap = self._predict_domain(query, tags, technologies)

        search_domains, dual = self._get_search_domains(domain, conf)

        # 2 — Vecteur requête
        q_vec = self._query_vector(query, tags, technologies)

        # 3 — Recherche
        n_search = min(self.top_k * 8, len(self.df))

        if self._use_faiss and self.index is not None:
            distances, indices = self.index.search(q_vec, n_search)
            dists   = distances[0]
            idxs    = indices[0]
        else:
            sims  = cosine_similarity(q_vec, self.X_norm)[0]
            idxs  = np.argsort(sims)[::-1][:n_search]
            dists = sims[idxs]

        # 4 — Filtrage + construction résultats
        results = []
        seen: set[str] = set()
        rank = 1

        for dist, idx in zip(dists, idxs):
            if idx < 0 or idx >= len(self.df):
                continue

            row    = self.df.iloc[idx]
            rdom   = str(row.get("domain") or "Other")

            # Filtre domaine
            if domain != "Unknown" and rdom not in search_domains:
                continue

            key = str(row.get("full_name") or row.get("name") or idx)
            if key in seen:
                continue
            seen.add(key)

            score = float(dist)
            t     = DOMAIN_THRESHOLDS.get(rdom, DEFAULT_THRESHOLDS)
            if score >= t["very_similar"]:
                sim_label = "very_similar"
            elif score >= t["similar"]:
                sim_label = "similar"
            else:
                sim_label = "related"

            results.append(SimilarProject(
                rank             = rank,
                name             = str(row.get("name") or ""),
                full_name        = str(row.get("full_name") or ""),
                description      = str(row.get("description") or "")[:180],
                domain           = rdom,
                author           = str(row.get("author") or ""),
                url              = str(row.get("url") or ""),
                stars            = int(row.get("stars") or 0),
                has_paper        = bool(row.get("has_paper")),
                is_deployed      = bool(row.get("is_deployed")),
                technologies     = row.get("techs_list") or [],
                score            = score,
                similarity_label = sim_label,
                same_domain      = rdom == domain,
            ))
            rank += 1

            if len(results) >= self.top_k:
                break

        # Fallback si trop peu de résultats
        if len(results) < 3 and domain != "Unknown":
            for dist, idx in zip(dists, idxs):
                if idx < 0 or idx >= len(self.df):
                    continue
                row = self.df.iloc[idx]
                key = str(row.get("full_name") or row.get("name") or idx)
                if key in seen:
                    continue
                seen.add(key)
                rdom  = str(row.get("domain") or "Other")
                score = float(dist)
                t     = DOMAIN_THRESHOLDS.get(rdom, DEFAULT_THRESHOLDS)
                results.append(SimilarProject(
                    rank             = rank,
                    name             = str(row.get("name") or ""),
                    full_name        = str(row.get("full_name") or ""),
                    description      = str(row.get("description") or "")[:180],
                    domain           = rdom,
                    author           = str(row.get("author") or ""),
                    url              = str(row.get("url") or ""),
                    stars            = int(row.get("stars") or 0),
                    has_paper        = bool(row.get("has_paper")),
                    is_deployed      = bool(row.get("is_deployed")),
                    technologies     = row.get("techs_list") or [],
                    score            = score,
                    similarity_label = (
                        "very_similar" if score >= t["very_similar"]
                        else "similar" if score >= t["similar"]
                        else "related"
                    ),
                    same_domain      = False,
                ))
                rank += 1
                if len(results) >= self.top_k:
                    break

        # ═══════════════════════════════════════════════════════════════════
        # 5 — Métriques d'originalité (CORRIGÉE - plus claire et logique)
        # ═══════════════════════════════════════════════════════════════════
        
        max_sim = results[0].score if results else 0.0
        originality = round(1.0 - max_sim, 4)
        t = DOMAIN_THRESHOLDS.get(domain, DEFAULT_THRESHOLDS)
        
        # Calcul des seuils d'originalité (inversés par rapport aux seuils de similarité)
        # Exemple pour Computer Vision: t["original"]=0.33 → seuil_originalité = 0.67
        # Cela signifie: originalité >= 0.67 = "very_original"
        orig_very_similar_threshold = round(1.0 - t["very_similar"], 4)  # Seuil pour "similaire"
        orig_similar_threshold      = round(1.0 - t["similar"], 4)       # Seuil pour "original"
        orig_original_threshold     = round(1.0 - t["original"], 4)      # Seuil pour "very_original"
        
        # Logique claire : plus l'originalité est élevée, meilleur est le label
        if originality >= orig_original_threshold:
            orig_label = "very_original"
        elif originality >= orig_similar_threshold:
            orig_label = "original"
        elif originality >= orig_very_similar_threshold:
            orig_label = "similar"
        else:
            orig_label = "duplicate"

        dcounts: dict[str,int] = {}
        for r in results:
            dcounts[r.domain] = dcounts.get(r.domain, 0) + 1
        related = sorted(dcounts, key=lambda x: -dcounts[x])

        suggestions = self._suggestions(results, domain, conf, originality, pmap)

        return SimilarityResult(
            query             = query,
            predicted_domain  = domain,
            domain_confidence = conf,
            similar_projects  = results,
            related_domains   = related,
            originality_score = originality,
            originality_label = orig_label,
            suggestions       = suggestions,
            dual_search       = dual,
            search_domains    = search_domains,
        )

    def _suggestions(self, results, domain, conf, originality, pmap):
        tips = []
        t = DOMAIN_THRESHOLDS.get(domain, DEFAULT_THRESHOLDS)

        if originality >= (1 - t["original"]):
            tips.append(f"✅ Projet original dans '{domain}' (score {originality:.2f}).")
        elif originality >= (1 - t["similar"]):
            tips.append(f"⚡ Originalité modérée ({originality:.2f}) — différenciez l'architecture ou les données.")
        else:
            tips.append(f"⚠️ Très similaire à l'existant (originalité {originality:.2f}).")

        if conf < DUAL_SEARCH_CONFIDENCE and domain != "Unknown":
            ranked = sorted(pmap.items(), key=lambda x: -x[1])
            if len(ranked) >= 2:
                tips.append(
                    f"🔍 Domaine ambigu ({conf:.0%}) — "
                    f"aussi possible : '{ranked[1][0]}' ({ranked[1][1]:.0%})."
                )

        deployed = [r for r in results if r.is_deployed]
        if deployed:
            tips.append(f"📦 {len(deployed)} projet(s) similaire(s) déjà déployé(s).")

        with_paper = [r for r in results if r.has_paper]
        if with_paper:
            tips.append(f"📄 {len(with_paper)} projet(s) similaire(s) lié(s) à un paper.")

        all_t: list[str] = []
        for r in results[:5]:
            all_t.extend(r.technologies)
        if all_t:
            top = [t_ for t_, _ in Counter(all_t).most_common(5)]
            tips.append(f"🔧 Technologies dominantes : {', '.join(top)}")

        return tips

    def stats(self):
        if self.df is None:
            return {}

        domains = (
                self.df["domain"]
                .fillna("Other")
                .value_counts()
                .to_dict()
                if "domain" in self.df.columns
                else {}
            )

        sources = (
                self.df["source"]
                .fillna("unknown")
                .value_counts()
                .to_dict()
                if "source" in self.df.columns
                else {}
            )

        return {
                "total_projects": len(self.df),
                "domains": domains,
                "sources": sources,
                "index_dim": self.X_norm.shape[1] if self.X_norm is not None else 0,
                "uses_faiss": self._use_faiss,
            }

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="AI Similarity Engine (léger)")
    parser.add_argument("query",     help="Description du projet")
    parser.add_argument("--csv",     default="ai_projects.csv")
    parser.add_argument("--bundle",  default="xgboost_bundle.pkl")
    parser.add_argument("--tags",    default="")
    parser.add_argument("--tech",    default="")
    parser.add_argument("--domain",  default=None)
    parser.add_argument("--top-k",   type=int, default=10)
    args = parser.parse_args()

    engine = SimilarityEngineLight(top_k=args.top_k)
    engine.load(args.csv, args.bundle)
    result = engine.search(args.query, tags=args.tags, technologies=args.tech,
                           force_domain=args.domain)
    print(result)


if __name__ == "__main__":
    _cli()