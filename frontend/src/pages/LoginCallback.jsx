// frontend/src/pages/LoginCallback.jsx
// Page intermédiaire — reçoit token + user depuis GitHub OAuth callback
// URL : /login/callback?token=...&name=...&email=...&avatar=...

import { useEffect } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"

export default function LoginCallback({ onLogin }) {
  const [params]  = useSearchParams()
  const navigate  = useNavigate()

  useEffect(() => {
    const token  = params.get("token")
    const name   = params.get("name")  || ""
    const email  = params.get("email") || ""
    const avatar = params.get("avatar")|| ""

    if (!token) {
      navigate("/login?error=github_failed")
      return
    }

    const user = { email, name, avatar }
    localStorage.setItem("aiscope_token", token)
    localStorage.setItem("aiscope_user",  JSON.stringify(user))
    onLogin?.(user)
    navigate("/search")
  }, [])

  return (
    <div style={{
      display:"flex", alignItems:"center", justifyContent:"center",
      minHeight:"60vh", flexDirection:"column", gap:"16px"
    }}>
      <div className="spinner-ring" />
      <p style={{ color:"var(--muted)", fontSize:"14px" }}>
        Connexion GitHub en cours…
      </p>
    </div>
  )
}