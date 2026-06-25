// frontend/src/components/ProjectCard.jsx
const SIM = {
  very_similar: { label: "Très similaire", cls: "sim-red"    },
  similar:      { label: "Similaire",      cls: "sim-amber"  },
  related:      { label: "Connexe",        cls: "sim-teal"   },
}

export default function ProjectCard({ project }) {
  // Gestion sécurisée des champs optionnels
  const sim = project.similarity_label ? SIM[project.similarity_label] : null
  const pct = project.score ? Math.round(project.score * 100) : null

  return (
    <a href={project.url} target="_blank" rel="noopener noreferrer"
       className={`project-card ${sim ? sim.cls : ""}`}>
      
      {/* N'afficher le top que si on a des métriques */}
      {(project.rank || pct || sim) && (
        <div className="pc-top">
          {project.rank && <span className="pc-rank">#{project.rank}</span>}
          {pct && <span className="pc-score">{pct}%</span>}
          {sim && <span className={`pc-sim-tag ${sim.cls}`}>{sim.label}</span>}
        </div>
      )}
      
      <h3 className="pc-name" title={project.name}>{project.name || "Sans titre"}</h3>
      {project.author && <p className="pc-author">by {project.author}</p>}
      {project.description && <p className="pc-desc">{project.description}</p>}
      
      <div className="pc-footer">
        {project.domain && <span className="pc-domain">{project.domain}</span>}
        {project.stars > 0 && <span className="pc-stars">★ {project.stars.toLocaleString()}</span>}
        {project.has_paper && <span className="pc-badge paper">paper</span>}
        {project.is_deployed && <span className="pc-badge deployed">déployé</span>}
      </div>
      
      {Array.isArray(project.technologies) && project.technologies.length > 0 && (
        <div className="pc-techs">
          {project.technologies.slice(0, 4).map(t => (
            <span key={t} className="tech-chip">{t}</span>
          ))}
        </div>
      )}
    </a>
  )
}