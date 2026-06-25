"""
main.py — CLI complet : collecte incrémentale, export dataset, stats, diff

Commandes :
    python main.py collect                    # collecte complète (incrémentale)
    python main.py collect --reset            # repart de zéro (garde les données)
    python main.py collect --github-only
    python main.py collect --hf-only
    python main.py collect --max-hf-models 1000

    python main.py export                     # → CSV + JSON + Parquet
    python main.py export --format parquet
    python main.py export --format csv
    python main.py export --format json
    python main.py export --format hf         # upload vers HuggingFace Hub

    python main.py stats                      # statistiques détaillées
    python main.py diff                       # checkpoint actuel
    python main.py enrich                     # remplit les domaines manquants
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from models import init_db, db_session, AIProject
from collector import run_collection, Checkpoint
from utils import infer_domain

log = logging.getLogger("main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

EXPORT_DIR = Path("dataset_exports")


# ─────────────────────────────────────────────────────────────────────────────
# Export CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(path: str = None) -> Path:
    """
    Export CSV par batch de 500 lignes — ne charge jamais tout en RAM.
    Lisible directement dans Excel, pandas, tout tableur.
    """
    EXPORT_DIR.mkdir(exist_ok=True)
    out     = Path(path) if path else EXPORT_DIR / "ai_projects.csv"
    session = db_session()

    # Colonnes dans l'ordre défini par to_dict()
    sample = session.query(AIProject).first()
    if sample is None:
        session.close()
        print("⚠️  Base vide.")
        return out

    fieldnames = list(sample.to_dict().keys())
    BATCH = 500
    total = 0

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_NONNUMERIC,   # force les strings entre guillemets → plus robuste
        )
        writer.writeheader()

        offset = 0
        while True:
            batch = (
                session.query(AIProject)
                .order_by(AIProject.id)
                .offset(offset)
                .limit(BATCH)
                .all()
            )
            if not batch:
                break
            for p in batch:
                row = p.to_dict()
                # Sérialise les listes JSON en string lisible (pas de [ " ] dans les cellules)
                for col in ("tags", "technologies"):
                    val = row.get(col)
                    if isinstance(val, list):
                        row[col] = "; ".join(val)
                writer.writerow(row)
            total   += len(batch)
            offset  += BATCH
            if len(batch) < BATCH:
                break

    session.close()
    size_kb = out.stat().st_size / 1024
    print(f"✅ CSV  → {out}  ({total:,} lignes, {size_kb:,.0f} KB)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Export JSON
# ─────────────────────────────────────────────────────────────────────────────

def export_json(path: str = None) -> Path:
    EXPORT_DIR.mkdir(exist_ok=True)
    out = Path(path) if path else EXPORT_DIR / "ai_projects.json"
    session  = db_session()
    projects = session.query(AIProject).all()
    data     = [p.to_dict() for p in projects]
    session.close()

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON → {out}  ({len(data)} entrées)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Export Parquet (format dataset standard — pandas / HuggingFace datasets)
# ─────────────────────────────────────────────────────────────────────────────

def export_parquet(path: str = None) -> Path:
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas non installé — pip install pandas pyarrow")
        sys.exit(1)

    EXPORT_DIR.mkdir(exist_ok=True)
    out = Path(path) if path else EXPORT_DIR / "ai_projects.parquet"

    session  = db_session()
    projects = session.query(AIProject).all()
    session.close()

    if not projects:
        print("⚠️  Base vide.")
        return out

    rows = [p.to_dict() for p in projects]
    df   = pd.DataFrame(rows)

    # Typage propre
    bool_cols = ["is_deployed", "has_demo", "is_fork", "is_archived",
                 "has_paper", "has_tests", "has_ci", "has_docker",
                 "has_requirements", "has_dataset"]
    int_cols  = ["stars", "forks", "watchers", "open_issues", "downloads"]
    float_cols = ["popularity_score"]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")

    size_mb = out.stat().st_size / 1_048_576
    print(f"✅ Parquet → {out}  ({len(df):,} lignes, {len(df.columns)} colonnes, {size_mb:.2f} MB)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Export complet (tous les formats à la fois)
# ─────────────────────────────────────────────────────────────────────────────

def export_all():
    print("\n📦 Export multi-format...")
    export_csv()
    export_json()
    export_parquet()
    print(f"\n📁 Fichiers dans : {EXPORT_DIR.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Upload vers HuggingFace Hub (dataset public/privé)
# ─────────────────────────────────────────────────────────────────────────────

def export_to_hf_hub(repo_id: str, private: bool = False):
    try:
        from datasets import Dataset
        import pandas as pd
    except ImportError:
        print("❌  pip install datasets pandas pyarrow")
        sys.exit(1)

    session  = db_session()
    projects = session.query(AIProject).all()
    session.close()

    rows = [p.to_dict() for p in projects]
    df   = pd.DataFrame(rows)
    ds   = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, private=private)
    print(f"✅ Dataset uploadé → https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistiques
# ─────────────────────────────────────────────────────────────────────────────

def print_stats():
    from sqlalchemy import func

    session = db_session()
    total   = session.query(AIProject).count()

    if total == 0:
        print("⚠️  Base vide — lancez d'abord: python main.py collect")
        session.close()
        return

    by_source  = session.query(AIProject.source, func.count()).group_by(AIProject.source).all()
    by_domain  = session.query(AIProject.domain, func.count()).group_by(AIProject.domain).all()
    deployed   = session.query(AIProject).filter_by(is_deployed=True).count()
    with_demo  = session.query(AIProject).filter_by(has_demo=True).count()
    archived   = session.query(AIProject).filter_by(is_archived=True).count()
    with_paper = session.query(AIProject).filter_by(has_paper=True).count()
    no_domain  = session.query(AIProject).filter(AIProject.domain.is_(None)).count()
    session.close()

    w = 55
    print(f"\n{'═'*w}")
    print(f"  📊 BASE AI PROJECTS — STATISTIQUES")
    print(f"{'═'*w}")
    print(f"  {'Total projets':<25} {total:>8,}")
    print(f"  {'Déployés':<25} {deployed:>8,}  ({deployed/total*100:.1f}%)")
    print(f"  {'Avec démo':<25} {with_demo:>8,}  ({with_demo/total*100:.1f}%)")
    print(f"  {'Archivés':<25} {archived:>8,}  ({archived/total*100:.1f}%)")
    print(f"  {'Avec paper':<25} {with_paper:>8,}  ({with_paper/total*100:.1f}%)")
    print(f"  {'Sans domaine':<25} {no_domain:>8,}  (→ lancer enrich)")
    print(f"\n  {'─'*45}")
    print(f"  Par source :")
    for src, count in sorted(by_source, key=lambda x: -x[1]):
        bar = "█" * int(count / total * 30)
        print(f"    {src:<18} {count:>6,}  {bar}")
    print(f"\n  Par domaine (top 10) :")
    top_domains = sorted(
        [(d or "Unknown", c) for d, c in by_domain],
        key=lambda x: -x[1]
    )[:10]
    for domain, count in top_domains:
        bar = "█" * int(count / total * 25)
        print(f"    {domain:<28} {count:>5,}  {bar}")
    print(f"{'═'*w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Enrichissement domaines
# ─────────────────────────────────────────────────────────────────────────────

def enrich_domains(force: bool = False):
    """
    Remplit ou recalcule le domaine de tous les projets.
    - Par défaut : remplit seulement les projets sans domaine (domain IS NULL).
    - force=True : recalcule TOUS les projets, même ceux déjà classés.
    """
    session  = db_session()
    query    = session.query(AIProject)
    if not force:
        query = query.filter(AIProject.domain.is_(None))

    projects = query.all()
    updated  = 0
    for p in projects:
        tags         = json.loads(p.tags) if p.tags else []
        pipeline_tag = p.pipeline_tag or ""
        new_domain   = infer_domain(p.description or "", tags, pipeline_tag=pipeline_tag)
        if new_domain != p.domain:
            p.domain = new_domain
            updated += 1

    session.commit()
    session.close()
    scope = "tous les projets" if force else "projets sans domaine"
    print(f"✅ {updated} projets mis à jour ({scope}).")


# ─────────────────────────────────────────────────────────────────────────────
# Diff / checkpoint info
# ─────────────────────────────────────────────────────────────────────────────

def print_diff():
    cp = Checkpoint()
    summary = cp.summary()
    print(f"\n📌 Checkpoint actuel ({Checkpoint().path}) :")
    for src, count in summary.items():
        status = "✅ complet" if count > 0 else "⬜ pas encore traité"
        print(f"  {src:<15} {count:>3} étapes terminées  {status}")

    session = db_session()
    from sqlalchemy import func
    by_source = session.query(AIProject.source, func.count()).group_by(AIProject.source).all()
    session.close()
    print(f"\n  Données en base :")
    for src, count in sorted(by_source, key=lambda x: -x[1]):
        print(f"  {src:<18} {count:>6,} projets")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_db()
    parser = argparse.ArgumentParser(
        description="AI Project Collector — Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # collect ──────────────────────────────────────────────────────────────
    c = sub.add_parser("collect", help="Collecte incrémentale (reprend où elle s'est arrêtée)")
    c.add_argument("--reset", action="store_true",
                   help="Réinitialise le checkpoint (re-vérifie tout, sans effacer la DB)")
    c.add_argument("--github-only",  action="store_true")
    c.add_argument("--hf-only",      action="store_true")
    c.add_argument("--pwc-only",     action="store_true")
    c.add_argument("--arxiv-only",   action="store_true")
    c.add_argument("--kaggle-only",  action="store_true")
    c.add_argument("--zenodo-only",  action="store_true")
    c.add_argument("--openml-only",  action="store_true")
    c.add_argument("--no-pwc",       action="store_true")
    c.add_argument("--no-arxiv",     action="store_true")
    c.add_argument("--no-kaggle",    action="store_true")
    c.add_argument("--no-zenodo",    action="store_true")
    c.add_argument("--no-openml",    action="store_true")
    c.add_argument("--max-github",    type=int, default=100)
    c.add_argument("--max-hf-models", type=int, default=500)
    c.add_argument("--max-hf-spaces", type=int, default=200)
    c.add_argument("--max-pwc",       type=int, default=200)
    c.add_argument("--max-arxiv",     type=int, default=500)
    c.add_argument("--max-kaggle",    type=int, default=100)
    c.add_argument("--max-zenodo",    type=int, default=200)
    c.add_argument("--max-openml",    type=int, default=500)

    # export ───────────────────────────────────────────────────────────────
    e = sub.add_parser("export", help="Exporte le dataset")
    e.add_argument("--format",
                   choices=["csv", "json", "parquet", "all", "hf"],
                   default="all")
    e.add_argument("--output", default=None, help="Chemin de sortie (optionnel)")
    e.add_argument("--hf-repo", default=None,
                   help="HuggingFace repo ID pour --format hf (ex: username/ai-projects)")
    e.add_argument("--private", action="store_true",
                   help="Dataset privé sur HuggingFace Hub")

    # stats ────────────────────────────────────────────────────────────────
    sub.add_parser("stats", help="Statistiques de la base")

    # diff ─────────────────────────────────────────────────────────────────
    sub.add_parser("diff", help="État du checkpoint et données en base")

    # enrich ───────────────────────────────────────────────────────────────
    en = sub.add_parser("enrich", help="Recalcule domaines et papers")
    en.add_argument("--force", action="store_true",
                    help="Recalcule TOUS les projets, même déjà classés")

    args = parser.parse_args()

    if args.command == "collect":
        only_flags = ["pwc_only","arxiv_only","github_only","hf_only",
                      "kaggle_only","zenodo_only","openml_only"]
        any_only   = any(getattr(args, f, False) for f in only_flags)

        def src(name):
            only_flag = f"{name}_only"
            no_flag   = f"no_{name}"
            if any_only:
                return getattr(args, only_flag, False)
            return not getattr(args, no_flag, False)

        run_collection(
            github   = src("github"),
            hf_rss   = src("hf") or (not any_only and not getattr(args,"no_hf",False)),
            hf_models= src("hf") or (not any_only and not getattr(args,"no_hf",False)),
            hf_spaces= src("hf") or (not any_only and not getattr(args,"no_hf",False)),
            pwc      = src("pwc"),
            arxiv    = src("arxiv"),
            kaggle   = src("kaggle"),
            zenodo   = src("zenodo"),
            openml   = src("openml"),
            max_github    = args.max_github,
            max_hf_models = args.max_hf_models,
            max_hf_spaces = args.max_hf_spaces,
            max_pwc       = args.max_pwc,
            max_arxiv     = args.max_arxiv,
            max_kaggle    = args.max_kaggle,
            max_zenodo    = args.max_zenodo,
            max_openml    = args.max_openml,
            reset_checkpoint = args.reset,
        )
        enrich_domains()
        print_stats()

    elif args.command == "export":
        fmt = args.format
        if fmt == "csv":
            export_csv(args.output)
        elif fmt == "json":
            export_json(args.output)
        elif fmt == "parquet":
            export_parquet(args.output)
        elif fmt == "hf":
            repo = args.hf_repo or input("HuggingFace repo ID (ex: username/ai-projects): ")
            export_to_hf_hub(repo, private=args.private)
        else:
            export_all()

    elif args.command == "stats":
        print_stats()

    elif args.command == "diff":
        print_diff()

    elif args.command == "enrich":
        enrich_domains(force=args.force)
        print_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()