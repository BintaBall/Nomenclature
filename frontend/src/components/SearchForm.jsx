// frontend/src/components/SearchForm.jsx
import { useState } from "react"

const DOMAINS = [
  "Computer Vision", "NLP / Text", "Audio / Speech", "Generative AI",
  "Medical / Healthcare", "Robotics / RL", "Multimodal", "Time Series",
  "Finance", "3D / Point Cloud", "Graph / Network", "Other",
]

export default function SearchForm({ onSearch, loading }) {
  const [desc,   setDesc]   = useState("")
  const [tags,   setTags]   = useState("")
  const [tech,   setTech]   = useState("")
  const [domain, setDomain] = useState("")
  const [open,   setOpen]   = useState(false)

  const ready = desc.trim().length >= 20

  const submit = (e) => {
    e.preventDefault()
    if (!ready) return
    onSearch({ description: desc.trim(), tags, technologies: tech, force_domain: domain || null, top_k: 10 })
  }

  return (
    <form className="search-form" onSubmit={submit}>
      <div className="sf-main">
        <label className="sf-label">Description du projet</label>
        <textarea
          className="sf-textarea"
          rows={5}
          maxLength={2000}
          placeholder="Ex: Système de détection de tumeurs cérébrales sur IRM en utilisant un transformer avec mécanisme d'attention, entraîné sur BraTS 2023..."
          value={desc}
          onChange={e => setDesc(e.target.value)}
        />
        <div className="sf-char">
          <span className={desc.length < 20 ? "char-warn" : "char-ok"}>
            {desc.length}/2000{desc.length < 20 && ` — minimum 20`}
          </span>
        </div>
      </div>

      <div className="sf-toggle-row">
        <button type="button" className="sf-toggle" onClick={() => setOpen(o => !o)}>
          {open ? "▲" : "▼"} Options avancées
        </button>
      </div>

      {open && (
        <div className="sf-advanced">
          <div className="sf-field">
            <label>Tags <span className="sf-hint">séparés par ;</span></label>
            <input value={tags} onChange={e => setTags(e.target.value)}
              placeholder="medical-imaging; segmentation; brain-tumor" />
          </div>
          <div className="sf-field">
            <label>Technologies <span className="sf-hint">séparées par ;</span></label>
            <input value={tech} onChange={e => setTech(e.target.value)}
              placeholder="PyTorch; HuggingFace Transformers" />
          </div>
          <div className="sf-field">
            <label>Forcer un domaine <span className="sf-hint">optionnel</span></label>
            <select value={domain} onChange={e => setDomain(e.target.value)}>
              <option value="">Auto-détection (recommandé)</option>
              {DOMAINS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
        </div>
      )}

      <button type="submit" className={`sf-submit ${!ready || loading ? "disabled" : ""}`}
        disabled={!ready || loading}>
        {loading ? <span className="sf-spinner">⟳  Analyse en cours…</span> : "Analyser le projet →"}
      </button>
    </form>
  )
}
