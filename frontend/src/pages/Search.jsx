// frontend/src/pages/Search.jsx
import { useState } from "react"
import { searchSimilarity, searchExternal, saveToHistory, contributeToDataset } from "../services/api"
import SearchForm from "../components/SearchForm"
import ResultsPanel from "../components/ResultsPanel"

export default function Search() {
  const [results, setResults]     = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [savedId, setSavedId]     = useState(null)
  const [contributed, setContributed] = useState(false)
  const [lastPayload, setLastPayload] = useState(null)

  const handleSearch = async (formData) => {
    setLoading(true)
    setError(null)
    setResults(null)
    setSavedId(null)
    setContributed(false)
    setLastPayload(formData)
    try {
      const data = await searchSimilarity(formData)
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // ✅ Recherche externe déclenchée depuis Search (pas depuis ResultsPanel)
  const handleExternal = async () => {
    if (!lastPayload) return
    
    setResults(prev => ({ ...prev, external_loading: true }))
    try {
      const ext = await searchExternal({
        description:  lastPayload.description,
        tags:         lastPayload.tags,
        technologies: lastPayload.technologies,
        domain:       results?.predicted_domain || "",
      })
      setResults(prev => ({ 
        ...prev, 
        external_triggered: true, 
        external_results: ext.results,
        external_loading: false
      }))
    } catch (e) {
      setError(e.message)
      setResults(prev => ({ ...prev, external_loading: false }))
    }
  }

  const handleSave = async () => {
    if (!results || !lastPayload) return
    try {
      const res = await saveToHistory({
        description:        lastPayload.description,
        tags:               lastPayload.tags,
        technologies:       lastPayload.technologies,
        predicted_domain:   results.predicted_domain,
        domain_confidence:  results.domain_confidence,
        originality_score:  results.originality_score,
        originality_label:  results.originality_label,
        similar_projects:   results.similar_projects?.slice(0, 5) || [],
      })
      setSavedId(res.id || null)
    } catch (e) {
      setError(e.message)
    }
  }

  const handleContribute = async () => {
    if (!results || !lastPayload) return
    try {
      await contributeToDataset({
        description:        lastPayload.description,
        tags:               lastPayload.tags,
        technologies:       lastPayload.technologies,
        predicted_domain:   results.predicted_domain,
        originality_score:  results.originality_score,
        originality_label:  results.originality_label,
        submission_id:      savedId,
      })
      setContributed(true)
    } catch (e) {
      setError(e.message)
    }
  }

  const canContribute = results && results.originality_score >= 0.45
  const saved = savedId !== null
  const extLoading = results?.external_loading || false
  const externalDone = results?.external_triggered || false

  return (
    <div className="search-page">
      <div className="search-page-header">
        <h1>Analyser un projet</h1>
        <p className="page-sub">
          Décrivez votre projet — AIScope le compare à {" "}
          <span className="highlight">23 000+ projets IA</span> et calcule son originalité.
        </p>
      </div>

      <SearchForm onSearch={handleSearch} loading={loading} />

      {error && <div className="error-banner">⚠ {error}</div>}

      {results && (
        <>
          {/* ✅ ResultsPanel ne reçoit que results */}
          <ResultsPanel results={results} />

          {/* ── 3 boutons d'action ── */}
          <div className="action-bar">

            <button
              className={`action-btn btn-external ${extLoading ? "loading" : ""}`}
              onClick={handleExternal}
              disabled={extLoading || externalDone}
              title="Rechercher sur GitHub, HuggingFace et ArXiv"
            >
              {extLoading ? "⟳ Recherche..." : externalDone ? "✓ Externe fait" : "🌐 Recherche externe"}
            </button>

            <button
              className={`action-btn btn-save ${saved ? "done" : ""}`}
              onClick={handleSave}
              disabled={saved}
              title="Sauvegarder dans votre historique"
            >
              {saved ? "✓ Sauvegardé" : "💾 Sauvegarder"}
            </button>

            <div className="contribute-wrapper">
              <button
                className={`action-btn btn-contribute ${contributed ? "done" : ""} ${!canContribute ? "locked" : ""}`}
                onClick={handleContribute}
                disabled={contributed || !canContribute}
                title={
                  !canContribute
                    ? `Score trop faible (${(results.originality_score * 100).toFixed(0)}% < 45%) pour contribuer`
                    : "Ajouter ce projet au dataset public"
                }
              >
                {contributed ? "✓ Contribué" : "📚 Contribuer au dataset"}
              </button>
              {!canContribute && (
                <span className="contribute-hint">
                  Score {(results.originality_score * 100).toFixed(0)}% — minimum 45% requis
                </span>
              )}
            </div>

          </div>
        </>
      )}
    </div>
  )
}