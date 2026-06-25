// frontend/src/services/api.js
// Toutes les requêtes API centralisées ici — propulsé par Axios
import axios from "axios"

// ── Instance Axios configurée ─────────────────────────────────────────────
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "https://nomenclature.glybette.com/api",
  timeout: 60000,   // 60s — la recherche externe + proxy peut être lente
  headers: { "Content-Type": "application/json" },
})

// Intercepteur requêtes
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("aiscope_token")
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
  },
  (error) => Promise.reject(error)
)

// Intercepteur réponses
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const msg =
      error.code === "ECONNABORTED"
        ? "Délai dépassé — le serveur met trop de temps à répondre (première analyse plus longue)"
        : error.response?.data?.detail ||
          error.response?.data?.message ||
          error.message ||
          "Erreur réseau"
    return Promise.reject(new Error(msg))
  }
)

// ── Endpoints ─────────────────────────────────────────────────────────────

// /search — timeout 60s (première fois : build index ~6s)
export const searchSimilarity = (payload) =>
  api.post("/search", payload, { timeout: 60000 })

// /external — timeout 90s (APIs externes + proxy lent)
export const searchExternal = (payload) =>
  api.post("/external", payload, { timeout: 90000 })

export const saveToHistory = (payload) =>
  api.post("/history", payload)

export const contributeToDataset = (payload) =>
  api.post("/contribute", payload)

export const getLibrary = (params = {}) =>
  api.get("/library", {
    params: Object.fromEntries(
      Object.entries(params).filter(([, v]) => v != null && v !== "")
    ),
  })

export const getStats = () => api.get("/stats")
export const getHistory = () => api.get("/history")

// ✅ AJOUT : Supprimer une soumission
export const deleteSubmission = (id) =>
  api.delete(`/submissions/${id}`)

// Si vous voulez aussi exporter toutes les fonctions en un seul objet
export default {
  searchSimilarity,
  searchExternal,
  saveToHistory,
  contributeToDataset,
  getLibrary,
  getStats,
  getHistory,
  deleteSubmission,
}