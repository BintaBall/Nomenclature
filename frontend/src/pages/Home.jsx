// frontend/src/pages/Home.jsx
import { useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { getStats } from "../services/api"

const FEATURES = [
  {
    icon: "◈",
    title: "Similarité interne",
    desc: "TF-IDF + FAISS sur 23 000 projets IA. Résultat en moins d'une seconde.",
    color: "teal",
  },
  {
    icon: "◎",
    title: "Score d'originalité",
    desc: "Seuils calibrés par domaine via analyse UMAP. Pas un chiffre générique.",
    color: "purple",
  },
  {
    icon: "◉",
    title: "Recherche externe",
    desc: "GitHub, HuggingFace et ArXiv si les résultats internes sont faibles.",
    color: "amber",
  },
  {
    icon: "◆",
    title: "Dataset collaboratif",
    desc: "Les projets originaux enrichissent la base. Tu contribues à la veille.",
    color: "blue",
  },
]

const DOMAINS = [
  "NLP / Text", "Computer Vision", "Robotics / RL", "Audio / Speech",
  "Generative AI", "Multimodal", "Medical / Healthcare", "Time Series",
  "Finance", "3D / Point Cloud", "Graph / Network", "Other",
]

export default function Home() {
  const navigate = useNavigate()
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getStats().then(setStats).catch(() => {})
  }, [])

  return (
    <div className="home">

      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-badge">Veille IA · 23 000+ projets indexés</div>
        <h1 className="hero-title">
          Votre projet IA<br />
          <span className="hero-accent">est-il original ?</span>
        </h1>
        <p className="hero-sub">
          Décrivez votre projet. AIScope le compare à toute la littérature IA connue
          et vous dit exactement où vous vous situez.
        </p>
        <div className="hero-actions">
          <button className="btn-primary" onClick={() => navigate("/search")}>
            Analyser mon projet →
          </button>
          <button className="btn-ghost" onClick={() => navigate("/library")}>
            Explorer la bibliothèque
          </button>
        </div>

        {stats && (
          <div className="hero-stats">
            {[
              [stats.total_projects?.toLocaleString(), "projets indexés"],
              [Object.keys(stats.domains || {}).length, "domaines IA"],
              [Object.keys(stats.sources || {}).length, "sources"],
              ["88%", "précision domaine"],
            ].map(([val, label]) => (
              <div className="hero-stat" key={label}>
                <span className="stat-val">{val}</span>
                <span className="stat-label">{label}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ── Features ── */}
      <section className="features">
        <h2 className="section-heading">Ce que fait AIScope</h2>
        <div className="features-grid">
          {FEATURES.map(f => (
            <div className={`feature-card feature-${f.color}`} key={f.title}>
              <span className="feature-icon">{f.icon}</span>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Domaines couverts ── */}
      <section className="domains-section">
        <h2 className="section-heading">12 domaines couverts</h2>
        <div className="domains-cloud">
          {DOMAINS.map(d => (
            <button
              key={d}
              className="domain-pill"
              onClick={() => navigate(`/library?domain=${encodeURIComponent(d)}`)}
            >
              {d}
            </button>
          ))}
        </div>
      </section>

      {/* ── CTA bas de page ── */}
      <section className="home-cta">
        <h2>Prêt à tester l'originalité de votre projet ?</h2>
        <button className="btn-primary btn-large" onClick={() => navigate("/search")}>
          Commencer l'analyse →
        </button>
      </section>

    </div>
  )
}
