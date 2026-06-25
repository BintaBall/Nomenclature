// frontend/src/pages/SubmissionDetail.jsx
import { useState, useEffect } from "react"
import { useParams, useNavigate, Link } from "react-router-dom"
import { getHistory } from "../services/api"
import { ORIG_LABELS } from "../components/ResultsPanel"
import OriginalityMeter from "../components/OriginalityMeter"

export default function SubmissionDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [submission, setSubmission] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchSubmission = async () => {
      try {
        setLoading(true)
        const data = await getHistory()
        // Chercher la soumission par ID (support de _id et id)
        const found = data.submissions?.find(s => s.id === id || s._id === id)
        if (found) {
          setSubmission(found)
        } else {
          setError("Soumission non trouvée")
        }
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchSubmission()
  }, [id])

  if (loading) {
    return (
      <div className="submission-detail-loading">
        <div className="spinner-ring" />
        <span>Chargement des détails...</span>
      </div>
    )
  }

  if (error || !submission) {
    return (
      <div className="submission-detail-error">
        <h2>Erreur</h2>
        <p>{error || "Soumission non trouvée"}</p>
        <button onClick={() => navigate("/library")} className="btn-primary">
          Retour à la bibliothèque
        </button>
      </div>
    )
  }

  const orig = ORIG_LABELS[submission.originality_label] || ORIG_LABELS.similar
  const pct = Math.round((submission.originality_score || 0) * 100)
  const date = submission.submitted_at
    ? new Date(submission.submitted_at).toLocaleDateString("fr-FR", {
        day: "2-digit",
        month: "long",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—"

  return (
    <div className="submission-detail">
      {/* En-tête */}
      <div className="submission-detail-header">
        <button onClick={() => navigate("/library")} className="back-btn">
          ← Retour
        </button>
        <h1>Détail de la soumission</h1>
        <span className={`detail-orig-badge ${orig.cls}`}>{orig.label}</span>
      </div>

      {/* Métriques principales */}
      <div className="submission-detail-metrics">
        <div className="metric-card">
          <span className="metric-label">Originalité</span>
          <OriginalityMeter score={submission.originality_score} />
          <span className="metric-value">{pct}%</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Domaine prédit</span>
          <span className="metric-value domain">{submission.predicted_domain || "Non défini"}</span>
          {submission.domain_confidence && (
            <span className="metric-confidence">
              Confiance : {Math.round(submission.domain_confidence * 100)}%
            </span>
          )}
        </div>
        <div className="metric-card">
          <span className="metric-label">Statut</span>
          <span className={`metric-status ${submission.is_public ? "public" : "private"}`}>
            {submission.is_public ? "✅ Contribué au dataset" : "🔒 Privé"}
          </span>
        </div>
      </div>

      {/* Description */}
      <div className="submission-detail-description">
        <h3>Description</h3>
        <p>{submission.description || "Aucune description"}</p>
      </div>

      {/* Tags et technologies */}
      <div className="submission-detail-metadata">
        {submission.tags && (
          <div className="meta-row">
            <span className="meta-label">Tags</span>
            <div className="meta-tags">
              {submission.tags.split(",").map((tag, i) => (
                <span key={i} className="meta-tag">{tag.trim()}</span>
              ))}
            </div>
          </div>
        )}
        {submission.technologies && (
          <div className="meta-row">
            <span className="meta-label">Technologies</span>
            <div className="meta-techs">
              {submission.technologies.split(",").map((tech, i) => (
                <span key={i} className="meta-tech">{tech.trim()}</span>
              ))}
            </div>
          </div>
        )}
        <div className="meta-row">
          <span className="meta-label">Soumis le</span>
          <span className="meta-value">{date}</span>
        </div>
        {submission.submitted_url && (
          <div className="meta-row">
            <span className="meta-label">URL source</span>
            <a href={submission.submitted_url} target="_blank" rel="noopener noreferrer" className="meta-link">
              {submission.submitted_url}
            </a>
          </div>
        )}
      </div>

      {/* Projets similaires détectés */}
      {submission.similar_projects && submission.similar_projects.length > 0 && (
        <div className="submission-detail-similars">
          <h3>
            Projets similaires détectés
            <span className="sim-count">{submission.similar_projects.length}</span>
          </h3>
          <div className="similars-list">
            {submission.similar_projects.map((proj, i) => (
              <div key={i} className="similar-item">
                <div className="similar-header">
                  <span className="similar-score">
                    {Math.round((proj.score || 0) * 100)}% match
                  </span>
                  <span className="similar-domain">{proj.domain || "Non défini"}</span>
                </div>
                <a
                  href={proj.url || "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="similar-name"
                >
                  {proj.name || "Sans titre"}
                </a>
                {proj.description && (
                  <p className="similar-desc">{proj.description.slice(0, 150)}...</p>
                )}
                {proj.stars > 0 && (
                  <span className="similar-stars">★ {proj.stars.toLocaleString()}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="submission-detail-actions">
        <Link to="/search" className="btn-primary">
          🔍 Analyser un nouveau projet
        </Link>
        {submission.is_public && (
          <span className="contribution-badge">
            ⬡ Contribué au dataset public
          </span>
        )}
      </div>
    </div>
  )
}