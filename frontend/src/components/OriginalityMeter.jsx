// frontend/src/components/OriginalityMeter.jsx

export default function OriginalityMeter({ score }) {
  // score = 1 - max_similarity (ex: 0.85 pour un projet très original)
  const pct = Math.round((score || 0) * 100)

  // Couleurs basées sur le score d'originalité
  // Vert si > 55% d'originalité, Rouge si < 25% (très similaire à l'existant)
  const getStyle = (s) => {
    if (s >= 0.55) return { color: "#16a34a", label: "Très original" };
    if (s >= 0.35) return { color: "#d97706", label: "Originalité modérée" };
    if (s >= 0.25) return { color: "#ea580c", label: "Assez similaire" };
    return { color: "#dc2626", label: "Déjà existant / Doublon" };
  };

  const { color, label } = getStyle(score);

  return (
    <div className="orig-meter" style={{ margin: "10px 0", width: "100%" }}>
      <div className="meter-info" style={{ display: "flex", justifyContent: "space-between", marginBottom: "5px" }}>
        <span className="meter-text" style={{ fontWeight: "bold", fontSize: "0.9rem" }}>Score d'originalité</span>
        <span className="meter-label" style={{ color, fontWeight: "bold" }}>
          {pct}% — {label}
        </span>
      </div>
      
      <div className="meter-track" style={{ background: "#e2e8f0", borderRadius: "10px", height: "12px", overflow: "hidden" }}>
        <div
          className="meter-fill"
          style={{ 
            width: `${pct}%`, 
            background: color,
            height: "100%",
            transition: "width 0.6s cubic-bezier(0.4, 0, 0.2, 1)"
          }}
        />
      </div>
    </div>
  )
}