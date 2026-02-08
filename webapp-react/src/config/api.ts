const urlParams = new URLSearchParams(window.location.search)

const FALLBACK_URL = 'http://localhost:8000'

const configuredUrl = urlParams.get('api') ||
                      import.meta.env?.VITE_API_URL ||
                      FALLBACK_URL

export const API_URL = configuredUrl
export { FALLBACK_URL }
