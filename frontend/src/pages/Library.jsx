// frontend/src/pages/Library.jsx
import { useState, useEffect, useCallback } from "react"
import { useSearchParams, useNavigate } from "react-router-dom"
import { getLibrary, getHistory, deleteSubmission } from "../services/api"
import { ProjectCard, ORIG_LABELS } from "../components/ResultsPanel"

const DOMAINS = [
  "Tous", "NLP / Text", "Computer Vision", "Robotics / RL", "Audio / Speech",
  "Generative AI", "Multimodal", "Medical / Healthcare", "Time Series",
  "Finance", "3D / Point Cloud", "Graph / Network", "Other",
]
const SOURCES = ["Toutes", "github", "arxiv", "hf_model", "pwc"]
const SORTS   = [
  { value: "popularity", label: "Popularité" },
  { value: "stars",      label: "Stars" },
  { value: "recent",     label: "Récents" },
]

// ── Carte soumission ──
function SubmissionCard({ item, onDelete }) {
  const navigate = useNavigate() // ✅ AJOUTER CETTE LIGNE
  const [expanded,  setExpanded]  = useState(false)
  const [deleting,  setDeleting]  = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  
  const orig = ORIG_LABELS[item.originality_label] || ORIG_LABELS.similar
  const pct  = Math.round((item.originality_score || 0) * 100)
  const date = item.submitted_at
    ? new Date(item.submitted_at).toLocaleDateString("fr-FR", {
        day: "2-digit", month: "short", year: "numeric",
      })
    : "—"

  const handleDelete = async () => {
    if (!confirmed) { setConfirmed(true); return }
    setDeleting(true)
    try {
      await deleteSubmission(item.id)
      onDelete(item.id)
    } catch (e) {
      alert(e.message)
      setDeleting(false)
      setConfirmed(false)
    }
  }

  return (
    <div className={`sub-card ${expanded ? "expanded" : ""}`}>
      <div className="sub-header" onClick={() => setExpanded(e => !e)}>
        <div className="sub-left">
          <span className={`sub-orig-badge ${orig.cls}`}>{orig.label}</span>
          <span className="sub-score">{pct}% original</span>
          {item.is_public && <span className="sub-public">contribué ⬡</span>}
        </div>
        <div className="sub-right">
          <span className="sub-domain">{item.predicted_domain}</span>
          <span className="sub-date">{date}</span>
          <span className="sub-expand">{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      <p className="sub-desc">
        {item.description?.slice(0, expanded ? 500 : 100)}
        {!expanded && item.description?.length > 100 && "…"}
      </p>

      {expanded && (
        <div className="sub-details">
          {item.tags && (
            <div className="sub-meta-row">
              <span className="sub-meta-label">Tags</span>
              <span className="sub-meta-val">{item.tags}</span>
            </div>
          )}
          {item.technologies && (
            <div className="sub-meta-row">
              <span className="sub-meta-label">Technologies</span>
              <span className="sub-meta-val">{item.technologies}</span>
            </div>
          )}
          <div className="sub-meta-row">
            <span className="sub-meta-label">Confiance domaine</span>
            <span className="sub-meta-val">
              {Math.round((item.domain_confidence || 0) * 100)}%
            </span>
          </div>

          {item.similar_projects?.length > 0 && (
            <div className="sub-similars">
              <p className="sub-similars-title">
                Projets similaires détectés ({item.similar_projects.length})
              </p>
              {item.similar_projects.map((p, i) => (
                <a
                  key={i}
                  href={p.url || "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="sub-sim-row"
                >
                  <span className="sub-sim-score">
                    {Math.round((p.score || 0) * 100)}%
                  </span>
                  <span className="sub-sim-name">{p.name}</span>
                  <span className="sub-sim-domain">{p.domain}</span>
                </a>
              ))}
            </div>
          )}

          <div className="sub-actions">
            {/* ✅ BOUTON VOIR LES DÉTAILS */}
            <button
              className="sub-detail-btn"
              onClick={(e) => {
                e.stopPropagation()
                navigate(`/library/submission/${item.id || item._id}`)
              }}
            >
              📄 Voir les détails complets
            </button>
            
            {/* Bouton de suppression */}
            <button
              className={`sub-del-btn ${confirmed ? "confirm" : ""}`}
              onClick={handleDelete}
              disabled={deleting}
            >
              {deleting   ? "⟳ Suppression…"
               : confirmed ? "⚠ Confirmer la suppression"
               : "🗑 Supprimer"}
            </button>
            {confirmed && (
              <button
                className="sub-cancel-btn"
                onClick={() => setConfirmed(false)}
              >
                Annuler
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function Library() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()

  const [tab, setTab] = useState("public")
  const [projects, setProjects] = useState([])
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)

  const [source, setSource] = useState("Toutes")
  const [sort, setSort] = useState("popularity")
  const [query, setQuery] = useState("")
  
  const domain = searchParams.get("domain") || "Tous"

  const fetchPublic = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getLibrary({
        dataset: tab === "contributed" ? "contributed" : "public",
        domain:  domain !== "Tous" ? domain : undefined,
        source:  source !== "Toutes" ? source : undefined,
        sort,
        q:       query || undefined,
        page,
        limit:   24,
      })
      setProjects(data.projects || [])
      setTotal(data.total || 0)
    } catch (e) {
      console.error("Erreur fetchPublic:", e)
      setProjects([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [tab, domain, source, sort, query, page])

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getHistory()
      setHistory(data.submissions || [])
    } catch (error) {
      console.error("Erreur fetchHistory:", error)
      setHistory([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let isMounted = true
    
    const loadData = async () => {
      if (!isMounted) return
      
      if (tab === "mine") {
        await fetchHistory()
      } else {
        await fetchPublic()
      }
    }
    
    loadData()
    
    return () => {
      isMounted = false
    }
  }, [tab, fetchPublic, fetchHistory])

  const handleDomain = (d) => {
    setPage(1)
    if (d !== "Tous") {
      setSearchParams({ domain: d })
    } else {
      setSearchParams({})
    }
  }

  const handleTabChange = (newTab) => {
    setTab(newTab)
    setPage(1)
    if (newTab === "mine") {
      setProjects([])
    } else {
      setHistory([])
    }
  }

  const handleDelete = (id) => {
    setHistory(prev => prev.filter(s => s.id !== id))
  }

  const user = (() => {
    try { return JSON.parse(localStorage.getItem("aiscope_user")) }
    catch { return null }
  })()

  return (
    <div className="library-page">

      <div className="library-header">
        <div className="lib-tabs">
          <button
            className={`lib-tab ${tab === "public" ? "active" : ""}`}
            onClick={() => handleTabChange("public")}
          >
            Dataset public
            <span className="tab-count">
              {total > 0 && tab === "public" ? total.toLocaleString() : "23k+"}
            </span>
          </button>
          <button
            className={`lib-tab ${tab === "contributed" ? "active" : ""}`}
            onClick={() => handleTabChange("contributed")}
          >
            Contributions
            <span className="tab-count">
              {total > 0 && tab === "contributed" ? total.toLocaleString() : "⬡"}
            </span>
          </button>
          <button
            className={`lib-tab ${tab === "mine" ? "active" : ""}`}
            onClick={() => handleTabChange("mine")}
          >
            Mes soumissions
            <span className="tab-count">{history.length}</span>
          </button>
        </div>

        {tab === "public" && (
          <div className="lib-search">
            <input
              type="text"
              placeholder="Rechercher dans la bibliothèque..."
              value={query}
              onChange={e => { setQuery(e.target.value); setPage(1) }}
              className="lib-search-input"
            />
          </div>
        )}
      </div>

      {(tab === "public" || tab === "contributed") && (
        <div className="lib-filters">
          <div className="filter-row">
            <span className="filter-label">Domaine</span>
            <div className="filter-pills">
              {DOMAINS.map(d => (
                <button
                  key={d}
                  className={`filter-pill ${domain === d ? "active" : ""}`}
                  onClick={() => handleDomain(d)}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>
          <div className="filter-row filter-row-sm">
            <div className="filter-group">
              <span className="filter-label">Source</span>
              <div className="filter-pills">
                {SOURCES.map(s => (
                  <button
                    key={s}
                    className={`filter-pill ${source === s ? "active" : ""}`}
                    onClick={() => { setSource(s); setPage(1) }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <div className="filter-group">
              <span className="filter-label">Trier par</span>
              <div className="filter-pills">
                {SORTS.map(s => (
                  <button
                    key={s.value}
                    className={`filter-pill ${sort === s.value ? "active" : ""}`}
                    onClick={() => { setSort(s.value); setPage(1) }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="lib-loading">
          <div className="spinner-ring" />
          <span>Chargement...</span>
        </div>
      ) : tab === "public" || tab === "contributed" ? (
        <>
          <div className="lib-meta">
            {total > 0 && (
              <span>
                {total.toLocaleString()} projets
                {domain !== "Tous" ? ` · ${domain}` : ""}
              </span>
            )}
          </div>
          <div className="lib-grid">
            {projects.length > 0 ? (
              projects.map((p, i) => (
                <ProjectCard key={p.id || p._id || i} project={p} />
              ))
            ) : (
              <div className="no-results">
                <p>Aucun projet trouvé</p>
              </div>
            )}
          </div>
          {total > 24 && (
            <div className="lib-pagination">
              <button
                className="page-btn"
                disabled={page === 1}
                onClick={() => setPage(p => Math.max(1, p - 1))}
              >
                ← Précédent
              </button>
              <span className="page-info">Page {page}</span>
              <button
                className="page-btn"
                disabled={projects.length < 24}
                onClick={() => setPage(p => p + 1)}
              >
                Suivant →
              </button>
            </div>
          )}
        </>
      ) : (
        <div className="submissions-section">
          {!user ? (
            <div className="submissions-empty">
              <p>Connectez-vous pour voir vos soumissions.</p>
              <button
                className="btn-primary"
                style={{ marginTop: "16px" }}
                onClick={() => navigate("/login")}
              >Se connecter</button>
            </div>
          ) : history.length === 0 ? (
            <div className="submissions-empty">
              <p>Aucune soumission pour l'instant.</p>
              <p>Analysez un projet pour le retrouver ici.</p>
              <button
                className="btn-primary"
                style={{ marginTop: "16px" }}
                onClick={() => navigate("/search")}
              >Analyser un projet</button>
            </div>
          ) : (
            <>
              <div className="submissions-meta">
                {history.length} soumission{history.length > 1 ? "s" : ""}
                <span className="sub-hint">
                  · Cliquez sur une carte pour voir les détails
                </span>
              </div>
              <div className="submissions-list">
                {history.map(item => (
                  <SubmissionCard
                    key={item.id || item._id}
                    item={item}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}