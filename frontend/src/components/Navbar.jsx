// frontend/src/components/Navbar.jsx
import { NavLink, useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { getStats } from "../services/api"

export default function Navbar({ user, onLogout, theme, onToggleTheme }) {
  const [stats, setStats] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    getStats().then(setStats).catch(() => {})
  }, [])

  return (
    <header className="navbar">
      <NavLink to="/" className="navbar-brand">
        <span className="brand-hex">⬡</span>
        <span className="brand-name">AIScope</span>
      </NavLink>

      <nav className="navbar-links">
        <NavLink to="/"         className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>Accueil</NavLink>
        <NavLink to="/search"   className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>Analyser</NavLink>
        <NavLink to="/library"  className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>Bibliothèque</NavLink>
        <NavLink to="/insights" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>Insights</NavLink>
      </nav>

      <div className="navbar-right">
        {stats && (
          <div className="navbar-stats">
            <span>{stats.total_projects?.toLocaleString()} projets</span>
            <span className="stat-dot">·</span>
            <span>{Object.keys(stats.domains || {}).length} domaines</span>
          </div>
        )}

        {/* Bouton toggle dark/light */}
        <button
          className="btn-theme"
          onClick={onToggleTheme}
          title={theme === "dark" ? "Passer en mode clair" : "Passer en mode sombre"}
          aria-label="Changer le thème"
        >
          {theme === "dark" ? "☀" : "☾"}
        </button>

        {user ? (
          <div className="navbar-user">
            {user.avatar && (
              <img src={user.avatar} alt={user.name} className="user-avatar" />
            )}
            <span className="user-name">{user.name || user.email}</span>
            <button className="btn-logout" onClick={onLogout}>Déconnexion</button>
          </div>
        ) : (
          <button className="btn-login" onClick={() => navigate("/login")}>
            Connexion
          </button>
        )}
      </div>
    </header>
  )
}