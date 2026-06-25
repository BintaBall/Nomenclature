"""
enrich.py — Enrichissement rétroactif des projets déjà en base.

À lancer UNE FOIS sur tes 16 248 projets existants, puis plus besoin
(les nouveaux projets collectés seront enrichis à la volée).

Ce que ça fait :
  1. Re-calcule le DOMAINE avec les nouvelles règles étendues
     (Generative AI, pipeline_tag, tags HF, etc.)
  2. Détecte les PAPERS ArXiv dans description + tags
  3. Met à jour has_paper, paper_url, domain en base
  4. Affiche un rapport avant/après

Usage :
    python enrich.py             # enrichit tout
    python enrich.py --dry-run  # affiche ce qui changerait, sans écrire
    python enrich.py --domain-only
    python enrich.py --paper-only
    python enrich.py --batch 1000   # taille de batch (défaut 500)
"""

import argparse
import json
import logging
import sys
from collections import Counter

from models import init_db, db_session, AIProject
from utils import infer_domain, extract_paper_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("enrich")

BATCH_SIZE = 500


# ─────────────────────────────────────────────────────────────────────────────
# Domaines
# ─────────────────────────────────────────────────────────────────────────────

def enrich_domains(session, dry_run: bool = False, batch_size: int = BATCH_SIZE) -> dict:
    """
    Re-calcule le domaine de TOUS les projets avec les nouvelles règles
    (pipeline_tag lookup, tags HF, texte étendu).
    Compare avant/après — n'écrit que si ça change.
    """
    log.info("═" * 55)
    log.info("ENRICHISSEMENT DOMAINES (recalcul complet)")
    log.info("═" * 55)

    total     = session.query(AIProject).count()
    changed   = 0
    unchanged = 0
    offset    = 0
    examples: list[tuple[str, str, str]] = []   # (name, old, new)

    before_counts: Counter = Counter()
    after_counts:  Counter = Counter()

    while True:
        batch = (
            session.query(AIProject)
            .order_by(AIProject.id)
            .offset(offset)
            .limit(batch_size)
            .all()
        )
        if not batch:
            break

        for p in batch:
            tags         = json.loads(p.tags) if p.tags else []
            pipeline_tag = p.pipeline_tag or ""
            old_domain   = p.domain or "Other"
            new_domain   = infer_domain(p.description or "", tags, pipeline_tag=pipeline_tag)

            before_counts[old_domain] += 1
            after_counts[new_domain]  += 1

            if new_domain != old_domain:
                if not dry_run:
                    p.domain = new_domain
                changed += 1
                if len(examples) < 10:
                    examples.append((p.name or p.full_name, old_domain, new_domain))
            else:
                unchanged += 1

        if not dry_run:
            session.commit()

        offset    += batch_size
        processed  = min(offset, total)
        if processed % 2000 == 0 or processed >= total:
            log.info(f"  {processed:>6,}/{total:,}  — changés={changed:,}  inchangés={unchanged:,}")

    log.info(f"\n  {'─'*45}")
    log.info(f"  Domaines AVANT (top 12):")
    for domain, count in before_counts.most_common(12):
        log.info(f"    {domain:<30} {count:>6,}")

    log.info(f"\n  Domaines APRÈS (top 12):")
    for domain, count in after_counts.most_common(12):
        log.info(f"    {domain:<30} {count:>6,}")

    if examples:
        log.info(f"\n  Exemples de reclassements :")
        for name, old, new in examples:
            log.info(f"    {name[:35]:<35}  {old}  →  {new}")

    log.info(f"\n  ✅ Changés   : {changed:,}")
    log.info(f"  ⏭  Inchangés : {unchanged:,}")
    if dry_run:
        log.info("  ⚠️  Mode DRY-RUN — rien n'a été écrit en base")

    return {"changed": changed, "unchanged": unchanged,
            "before": dict(before_counts), "after": dict(after_counts)}


def diagnose(session, n: int = 5):
    """
    Affiche n exemples de chaque source pour vérifier pipeline_tag, tags, domain.
    Utile pour debugger l'enrichissement.
    """
    log.info("═" * 55)
    log.info("DIAGNOSTIC")
    log.info("═" * 55)
    for source in ("hf_model", "hf_space", "github", "hf_rss"):
        samples = (
            session.query(AIProject)
            .filter_by(source=source)
            .limit(n)
            .all()
        )
        if not samples:
            continue
        log.info(f"\n  [{source}] — {n} exemples :")
        for p in samples:
            tags = json.loads(p.tags) if p.tags else []
            log.info(
                f"    name={p.name[:30]:<30} | "
                f"pipeline={p.pipeline_tag or '—':<25} | "
                f"domain={p.domain or '—':<20} | "
                f"tags={str(tags[:3])}"
            )
            # Recalcul live pour voir si ça changerait
            new = infer_domain(p.description or "", tags, pipeline_tag=p.pipeline_tag or "")
            if new != p.domain:
                log.info(f"      ⚡ CHANGERAIT → {new}")


# ─────────────────────────────────────────────────────────────────────────────
# Papers
# ─────────────────────────────────────────────────────────────────────────────

def enrich_papers(session, dry_run: bool = False, batch_size: int = BATCH_SIZE) -> dict:
    """
    Détecte les papers ArXiv/PDF dans description + tags de TOUS les projets.
    Met à jour has_paper et paper_url.
    """
    log.info("═" * 55)
    log.info("ENRICHISSEMENT PAPERS")
    log.info("═" * 55)

    total    = session.query(AIProject).count()
    found    = 0
    with_url = 0
    offset   = 0

    while True:
        batch = (
            session.query(AIProject)
            .order_by(AIProject.id)
            .offset(offset)
            .limit(batch_size)
            .all()
        )
        if not batch:
            break

        for p in batch:
            tags = json.loads(p.tags) if p.tags else []
            text = (p.description or "") + " " + (p.readme_excerpt or "")
            has_paper, paper_url = extract_paper_info(text, tags)

            if has_paper and not p.has_paper:
                if not dry_run:
                    p.has_paper  = True
                    p.paper_url  = paper_url
                found += 1
                if paper_url:
                    with_url += 1

        if not dry_run:
            session.commit()

        offset += batch_size
        processed = min(offset, total)
        log.info(f"  {processed:>6,}/{total:,}  — papers trouvés={found:,} (avec URL={with_url:,})")

    log.info(f"\n  ✅ Papers détectés : {found:,}")
    log.info(f"  🔗 Avec URL ArXiv  : {with_url:,}")
    if dry_run:
        log.info("  ⚠️  Mode DRY-RUN — rien n'a été écrit en base")

    return {"found": found, "with_url": with_url}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrichissement rétroactif des projets IA")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Simule sans écrire en base")
    parser.add_argument("--domain-only", action="store_true",
                        help="Enrichit seulement les domaines")
    parser.add_argument("--paper-only",  action="store_true",
                        help="Enrichit seulement les papers")
    parser.add_argument("--diagnose",    action="store_true",
                        help="Affiche des exemples pour debugger (sans écrire)")
    parser.add_argument("--batch",       type=int, default=BATCH_SIZE,
                        help=f"Taille de batch (défaut={BATCH_SIZE})")
    args = parser.parse_args()

    init_db()
    session = db_session()

    total = session.query(AIProject).count()
    log.info(f"Base : {total:,} projets")
    if total == 0:
        log.warning("Base vide — lancez d'abord: python main.py collect")
        sys.exit(0)

    if args.diagnose:
        diagnose(session)
        session.close()
        return

    do_domains = not args.paper_only
    do_papers  = not args.domain_only

    try:
        if do_domains:
            enrich_domains(session, dry_run=args.dry_run, batch_size=args.batch)

        if do_papers:
            enrich_papers(session, dry_run=args.dry_run, batch_size=args.batch)

    except KeyboardInterrupt:
        log.warning("Interruption — commit des changements en cours...")
        if not args.dry_run:
            session.commit()
    finally:
        session.close()

    log.info("\n✅ Enrichissement terminé.")


if __name__ == "__main__":
    main()