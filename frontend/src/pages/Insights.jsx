// frontend/src/pages/Insights.jsx
import { useEffect, useState } from "react"
import { getStats } from "../services/api"
import axios from "axios"

const BASE = import.meta.env.VITE_API_URL || "https://nomenclature.glybette.com/api"

function StatCard({ value, label, color = "teal" }) {
  return (
    <div className={`ins-stat-card ins-${color}`}>
      <div className="ins-val">{value}</div>
      <div className="ins-label">{label}</div>
    </div>
  )
}

function BarChart({ data, maxVal }) {
  return (
    <div className="ins-bar-list">
      {data.map(({ label, value, color }) => (
        <div key={label} className="ins-bar-row">
          <span className="ins-bar-label">{label}</span>
          <div className="ins-bar-track">
            <div
              className="ins-bar-fill"
              style={{ width: `${(value / maxVal) * 100}%`, background: color || "var(--teal)" }}
            />
          </div>
          <span className="ins-bar-val">{value.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

const ORIG_COLORS = {
  very_original: "#16a34a",
  original:      "#22c55e",
  similar:       "#d97706",
  duplicate:     "#dc2626",
}

const DOMAIN_COLORS = [
  "#0d9488","#2563eb","#7c3aed","#d97706",
  "#dc2626","#16a34a","#0891b2","#9333ea",
  "#c2410c","#0284c7","#15803d","#6d28d9",
]

export default function Insights() {
  const [stats,     setStats]     = useState(null)
  const [insights,  setInsights]  = useState(null)
  const [loading,   setLoading]   = useState(true)

  useEffect(() => {
    Promise.all([
      getStats(),
      axios.get(`${BASE}/insights`).then(r => r.data).catch(() => null),
    ]).then(([s, ins]) => {
      setStats(s)
      setInsights(ins)
      setLoading(false)
    })
  }, [])

  if (loading) return (
    <div className="ins-loading">
      <div className="spinner-ring" />
      <span>Chargement des données…</span>
    </div>
  )

  const domains = stats?.domains || {}
  const sources = stats?.sources || {}
  const totalProjects = stats?.total_projects || 0

  const domainData = Object.entries(domains)
    .sort(([,a],[,b]) => b - a)
    .map(([label, value], i) => ({ label, value, color: DOMAIN_COLORS[i % DOMAIN_COLORS.length] }))

  const sourceData = Object.entries(sources)
    .sort(([,a],[,b]) => b - a)
    .map(([label, value]) => ({ label, value }))

  const maxDomain = Math.max(...domainData.map(d => d.value), 1)
  const maxSource = Math.max(...sourceData.map(d => d.value), 1)

  // Données MongoDB (insights)
  const submissions    = insights?.total_submissions    || 0
  const contributions  = insights?.total_contributions || 0
  const avgScore       = insights?.avg_originality_score
  const origDist       = insights?.originality_distribution || {}
  const topDomains     = insights?.top_searched_domains || []

  return (
    <div className="insights-page">
      <div className="ins-header">
        <h1>Insights</h1>
        <p className="page-sub">Statistiques de la plateforme et de la base de données</p>
      </div>

      {/* ── Stats globales ── */}
      <section className="ins-section">
        <h2 className="ins-section-title">Dataset public</h2>
        <div className="ins-stats-grid">
          <StatCard value={totalProjects.toLocaleString()} label="Projets indexés"    color="teal"   />
          <StatCard value={domainData.length}              label="Domaines couverts"  color="blue"   />
          <StatCard value={sourceData.length}              label="Sources de collecte" color="purple" />
          <StatCard value="88%"                            label="Précision XGBoost"  color="green"  />
        </div>
      </section>

      {/* ── Distribution domaines ── */}
      <section className="ins-section">
        <h2 className="ins-section-title">Projets par domaine</h2>
        <div className="ins-chart-card">
          <BarChart data={domainData} maxVal={maxDomain} />
        </div>
      </section>

      {/* ── Distribution sources ── */}
      <section className="ins-section">
        <h2 className="ins-section-title">Sources de collecte</h2>
        <div className="ins-chart-card">
          <BarChart data={sourceData} maxVal={maxSource} />
        </div>
      </section>

      {/* ── Stats Dataset 2 (MongoDB) ── */}
      <section className="ins-section">
        <h2 className="ins-section-title">
          Activité de la plateforme
          <span className="ins-sub">données en temps réel — MongoDB</span>
        </h2>
        <div className="ins-stats-grid">
          <StatCard value={submissions.toLocaleString()}   label="Analyses effectuées"   color="teal"   />
          <StatCard value={contributions.toLocaleString()} label="Contributions publiques" color="blue"  />
          <StatCard
            value={avgScore != null ? `${(avgScore * 100).toFixed(0)}%` : "—"}
            label="Score originalité moyen"
            color="amber"
          />
          <StatCard
            value={Object.values(origDist).reduce((a,b) => a+b, 0).toLocaleString() || "—"}
            label="Projets classifiés"
            color="purple"
          />
        </div>
      </section>

      {/* ── Distribution scores originalité ── */}
      {Object.keys(origDist).length > 0 && (
        <section className="ins-section">
          <h2 className="ins-section-title">Distribution des scores d'originalité</h2>
          <div className="ins-chart-card">
            <div className="ins-orig-dist">
              {Object.entries(origDist).map(([label, count]) => {
                const total = Object.values(origDist).reduce((a,b)=>a+b,0) || 1
                const pct   = Math.round(count / total * 100)
                return (
                  <div key={label} className="ins-orig-row">
                    <span className="ins-orig-label">
                      <span
                        className="ins-orig-dot"
                        style={{ background: ORIG_COLORS[label] || "#888" }}
                      />
                      {label === "very_original" ? "Très original"
                       : label === "original"    ? "Original"
                       : label === "similar"     ? "Similaire"
                       : "Doublon"}
                    </span>
                    <div className="ins-bar-track" style={{ flex: 1 }}>
                      <div
                        className="ins-bar-fill"
                        style={{
                          width: `${pct}%`,
                          background: ORIG_COLORS[label] || "#888"
                        }}
                      />
                    </div>
                    <span className="ins-bar-val">{count} ({pct}%)</span>
                  </div>
                )
              })}
            </div>
          </div>
        </section>
      )}

      {/* ── Top domaines recherchés ── */}
      {topDomains.length > 0 && (
        <section className="ins-section">
          <h2 className="ins-section-title">Domaines les plus analysés</h2>
          <div className="ins-chart-card">
            <BarChart
              data={topDomains.map((d, i) => ({
                label: d._id, value: d.count,
                color: DOMAIN_COLORS[i % DOMAIN_COLORS.length]
              }))}
              maxVal={topDomains[0]?.count || 1}
            />
          </div>
        </section>
      )}

    </div>
  )
}