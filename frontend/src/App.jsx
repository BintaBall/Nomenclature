// frontend/src/App.jsx
import { BrowserRouter, Routes, Route } from "react-router-dom"
import { useState } from "react"
import useTheme from "./hooks/useTheme"
import Navbar from "./components/Navbar"
import Home from "./pages/Home"
import Search from "./pages/Search"
import Library from "./pages/Library"
import Login from "./pages/Login"
import LoginCallback from "./pages/LoginCallback"
import Insights from "./pages/Insights"
import SubmissionDetail from "./pages/SubmissionDetail" // 👈 Nouvel import

export default function App() {
  const { theme, toggle } = useTheme()

  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("aiscope_user")) }
    catch { return null }
  })

  const handleLogin  = (u) => setUser(u)
  const handleLogout = () => {
    localStorage.removeItem("aiscope_token")
    localStorage.removeItem("aiscope_user")
    setUser(null)
  }

  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar
          user={user}
          onLogout={handleLogout}
          theme={theme}
          onToggleTheme={toggle}
        />
        <main className="page-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/search" element={<Search />} />
            <Route path="/library" element={<Library />} />
            <Route path="/library/submission/:id" element={<SubmissionDetail />} /> {/* ✅ Nouvelle route */}
            <Route path="/insights" element={<Insights />} />
            <Route path="/login" element={<Login onLogin={handleLogin} />} />
            <Route path="/login/callback" element={<LoginCallback onLogin={handleLogin} />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}