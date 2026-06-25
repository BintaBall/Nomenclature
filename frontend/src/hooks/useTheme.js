// frontend/src/hooks/useTheme.js
import { useState, useEffect } from "react"

export default function useTheme() {
  // Lire la préférence sauvegardée, sinon utiliser la préférence système
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("aiscope_theme")
    if (saved) return saved
    return window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark"
  })

  useEffect(() => {
    // Appliquer le thème sur <html data-theme="...">
    document.documentElement.setAttribute("data-theme", theme)
    localStorage.setItem("aiscope_theme", theme)
  }, [theme])

  const toggle = () => setTheme(t => t === "dark" ? "light" : "dark")

  return { theme, toggle }
}