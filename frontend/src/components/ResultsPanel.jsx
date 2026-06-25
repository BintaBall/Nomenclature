// frontend/src/components/ResultsPanel.jsx

import OriginalityMeter from "./OriginalityMeter"

// Labels du BANDEAU principal (originality_score du projet soumis)
// originality_score = 1 - max_cosine → élevé = original, faible = doublon
const ORIG_STYLE = {
  very_original: { label: "Très original",  cls: "orig-green" },
  original:      { label: "Original",       cls: "orig-green" },
  similar:       { label: "Similaire",      cls: "orig-amber" },
  duplicate:     { label: "Déjà existant",  cls: "orig-red"   },
}

const SRC_ICON = {
  github: "★", huggingface: "🤗", huggingface_space: "▷",
  arxiv: "§", pwc: "◈", hf_model: "🤗", semantic_scholar: "📚",
}

// Labels sur les ProjectCards (originality_label des projets de la DB)
const ORIG_LABELS = {
  very_original: { label: "Très original", cls: "orig-tag-green" },
  original:      { label: "Original",      cls: "orig-tag-green" },
  similar:       { label: "Similaire",     cls: "orig-tag-amber" },
  duplicate:     { label: "Doublon",       cls: "orig-tag-red"   },
}

// similarity_label = à quel point ce projet DE LA DB ressemble à la requête
// → très similaire = score cosine élevé = projet existant proche
const SIM = {
  very_similar: { label: "Très similaire", cls: "sim-red"   },
  similar:      { label: "Similaire",      cls: "sim-amber" },
  related:      { label: "Connexe",        cls: "sim-teal"  },
}

export function ProjectCard({ project }) {
  const sim = project.similarity_label ? SIM[project.similarity_label] : SIM.related
  // score = cosine similarity avec la requête (élevé = très similaire)
  const pct = Math.round((project.score || 0) * 100)
  const orig = project.originality_label ? ORIG_LABELS[project.originality_label] : null
  const hasTech = Array.isArray(project.technologies) && project.technologies.length > 0

  return (
    <a
      href={project.url || "#"}
      target="_blank"
      rel="noopener noreferrer"
      className={`project-card ${sim.cls}`}
    >
      <div className="pc-top">
        {project.rank && <span className="pc-rank">#{project.rank}</span>}
        <span className="pc-score">{pct}% match</span>
        <span className={`pc-sim-tag ${sim.cls}`}>{sim.label}</span>
        {orig && <span className={`pc-orig-tag ${orig.cls}`}>{orig.label}</span>}
      </div>
      <h3 className="pc-name" title={project.name}>{project.name || "Sans titre"}</h3>
      {project.author && <p className="pc-author">by {project.author}</p>}
      <p className="pc-desc">{project.description || "Aucune description"}</p>
      <div className="pc-footer">
        {project.domain && <span className="pc-domain">{project.domain}</span>}
        {project.stars > 0 && <span className="pc-stars">★ {project.stars.toLocaleString()}</span>}
        {project.has_paper && <span className="pc-badge paper">paper</span>}
        {project.is_deployed && <span className="pc-badge deployed">déployé</span>}
      </div>
      {hasTech && (
        <div className="pc-techs">
          {project.technologies.slice(0, 4).map(t => (
            <span key={t} className="tech-chip">{t}</span>
          ))}
        </div>
      )}
    </a>
  )
}

// ✅ AJOUTER CETTE LIGNE POUR EXPORTER ORIG_LABELS
export { ORIG_LABELS }

export default function ResultsPanel({ results }) {
  const {
    predicted_domain,
    domain_confidence,
    originality_score,
    originality_label,
    dual_search,
    similar_projects = [],
    external_triggered,
    external_results = [],
    suggestions = [],
    processing_time_ms,
  } = results

  const orig = ORIG_STYLE[originality_label] || ORIG_STYLE.similar

  return (
    <div className="results-panel">

      {/* ── Bandeau résumé ── */}
      <div className="results-summary">
        <div className="rs-domain">
          <span className="rs-domain-name">{predicted_domain}</span>
          <span className="rs-conf">{(domain_confidence * 100).toFixed(0)}%</span>
          {dual_search && <span className="rs-dual">dual ↗</span>}
        </div>

        {/* Meter : originality_score élevé = original (barre verte longue) */}
        <OriginalityMeter score={originality_score} />

        {/* Badge label */}
        <span className={`rs-label ${orig.cls}`}>{orig.label}</span>

        {processing_time_ms && (
          <span className="rs-time">{processing_time_ms}ms</span>
        )}
      </div>

      {/* ── Suggestions ── */}
      {suggestions.length > 0 && (
        <div className="suggestions-box">
          {suggestions.map((s, i) => <p key={i}>{s}</p>)}
        </div>
      )}

      {/* ── Projets similaires internes ── */}
      {similar_projects.length > 0 && (
        <section className="results-section">
          <h2 className="rs-section-title">
            Projets similaires
            <span className="rs-count">{similar_projects.length}</span>
          </h2>
          <div className="results-grid">
            {similar_projects.map((p, i) => (
              <ProjectCard key={p.rank ?? p.id ?? i} project={p} />
            ))}
          </div>
        </section>
      )}

      {/* ── Résultats externes ── */}
      {external_triggered && external_results.length > 0 && (
        <section className="results-section ext-section">
          <h2 className="rs-section-title">
            Résultats externes
            <span className="rs-count">{external_results.length}</span>
            <span className="rs-source-label">HuggingFace · Semantic Scholar · PWC</span>
          </h2>
          <div className="ext-list">
            {external_results.map((r, i) => (
              <a
                key={i}
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                className="ext-item"
              >
                <span className="ext-src">{SRC_ICON[r.source] || "◉"}</span>
                <div className="ext-body">
                  <span className="ext-title">{r.title}</span>
                  <span className="ext-desc">{r.description?.slice(0, 110)}…</span>
                </div>
                <div className="ext-right">
                  {r.stars > 0 && (
                    <span className="ext-stars">★ {r.stars.toLocaleString()}</span>
                  )}
                  {r.has_paper && <span className="ext-paper">paper</span>}
                  <span className={`ext-source-tag src-${r.source?.replace("_", "-")}`}>
                    {r.source === "semantic_scholar" ? "S2"
                     : r.source === "huggingface" ? "HF"
                     : r.source === "huggingface_space" ? "Space"
                     : r.source?.toUpperCase?.() || r.source}
                  </span>
                  <span className="ext-rel">{(r.relevance * 100).toFixed(0)}%</span>
                </div>
              </a>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}