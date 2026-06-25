// frontend/src/pages/Login.jsx
import { useState } from "react"
import { useNavigate } from "react-router-dom"
import axios from "axios"

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"

export default function Login({ onLogin }) {
  const [tab,      setTab]      = useState("login")
  const [email,    setEmail]    = useState("")
  const [password, setPassword] = useState("")
  const [name,     setName]     = useState("")
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const navigate = useNavigate()

  const submit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const endpoint = tab === "register" ? "/auth/register" : "/auth/login"
      const payload  = tab === "register"
        ? { email, password, name }
        : { email, password }

      const res  = await axios.post(`${BASE}${endpoint}`, payload)
      const { token, user } = res.data

      localStorage.setItem("aiscope_token", token)
      localStorage.setItem("aiscope_user",  JSON.stringify(user))
      onLogin?.(user)
      navigate("/search")
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleGitHub = () => {
    // Redirige vers le backend qui redirige vers GitHub
    window.location.href = `${BASE}/auth/github`
  }

  // Lire l'erreur GitHub depuis l'URL si présente
  const urlError = new URLSearchParams(window.location.search).get("error")
  const ghError  = urlError === "github_denied"
    ? "Autorisation GitHub refusée."
    : urlError === "github_failed"
    ? "Erreur lors de la connexion GitHub. Réessayez."
    : null

  return (
    <div className="login-page">
      <div className="login-card">

        <div className="login-brand">
          <span className="brand-hex">⬡</span>
          <span className="brand-name">AIScope</span>
        </div>
        <p className="login-sub">
          {tab === "login"
            ? "Connectez-vous pour accéder à votre historique"
            : "Créez votre compte pour commencer"}
        </p>

        {/* GitHub OAuth */}
        <button className="btn-github" onClick={handleGitHub}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
          </svg>
          Continuer avec GitHub
        </button>

        <div className="login-divider">
          <span>ou</span>
        </div>

        {/* Tabs email/password */}
        <div className="login-tabs">
          <button
            className={`login-tab ${tab === "login" ? "active" : ""}`}
            onClick={() => { setTab("login"); setError(null) }}
          >Connexion</button>
          <button
            className={`login-tab ${tab === "register" ? "active" : ""}`}
            onClick={() => { setTab("register"); setError(null) }}
          >Créer un compte</button>
        </div>

        <form className="login-form" onSubmit={submit}>
          {tab === "register" && (
            <div className="lf-field">
              <label>Nom</label>
              <input
                type="text" value={name}
                onChange={e => setName(e.target.value)}
                placeholder="Votre nom" minLength={2}
              />
            </div>
          )}
          <div className="lf-field">
            <label>Email</label>
            <input
              type="email" value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="email@exemple.com" required
            />
          </div>
          <div className="lf-field">
            <label>Mot de passe</label>
            <input
              type="password" value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••" required minLength={6}
            />
          </div>

          {(error || ghError) && (
            <div className="lf-error">⚠ {error || ghError}</div>
          )}

          <button
            type="submit"
            className={`lf-submit ${loading ? "loading" : ""}`}
            disabled={loading}
          >
            {loading
              ? "⟳ Chargement…"
              : tab === "login" ? "Se connecter" : "Créer le compte"}
          </button>
        </form>

        <p className="login-skip">
          <button className="skip-btn" onClick={() => navigate("/search")}>
            Continuer sans compte →
          </button>
        </p>

      </div>
    </div>
  )
}