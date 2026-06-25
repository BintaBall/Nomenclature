// frontend/src/services/api.js
import axios from "axios"

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "https://nomenclature.glybette.com/api",
  timeout: 60000,
  headers: { "Content-Type": "application/json" },
})

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("aiscope_token")
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
  },
  (error) => Promise.reject(error)
)

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const msg =
      error.code === "ECONNABORTED"
        ? "Délai dépassé — le serveur met trop de temps à répondre"
        : error.response?.data?.detail ||
          error.response?.data?.message ||
          error.message ||
          "Erreur réseau"
    return Promise.reject(new Error(msg))
  }
)

export const searchSimilarity = (payload) =>
  api.post("/search", payload, { timeout: 60000 })

export const searchExternal = (payload) =>
  api.post("/external", payload, { timeout: 90000 })

export const saveToHistory = (payload) =>
  api.post("/history/save", payload)  // ✅ Corrigé

export const contributeToDataset = (payload) =>
  api.post("/contribute", payload)

export const getLibrary = (params = {}) =>
  api.get("/library", {
    params: Object.fromEntries(
      Object.entries(params).filter(([, v]) => v != null && v !== "")
    ),
  })

export const getStats = () => api.get("/stats")

export const getHistory = () => api.get("/history/mine")  // ✅ Corrigé

export const deleteSubmission = (id) =>
  api.delete(`/submissions/${id}`)

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