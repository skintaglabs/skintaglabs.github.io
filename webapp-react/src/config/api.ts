const urlParams = new URLSearchParams(window.location.search)

const configuredUrl = urlParams.get('api') ||
                      (import.meta as any).env?.VITE_API_URL ||
                      'http://localhost:8000'

export const API_URL = configuredUrl
export const FALLBACK_URL = 'http://localhost:8000'

// Try primary URL, fallback to localhost if it fails
export async function fetchWithFallback(
  endpoint: string,
  options?: RequestInit
): Promise<Response> {
  // Try configured URL first
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000)

    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      signal: controller.signal
    })

    clearTimeout(timeoutId)
    return response
  } catch (error) {
    // If configured URL fails and it's not already localhost, try localhost
    if (API_URL !== FALLBACK_URL) {
      console.warn(`API unreachable at ${API_URL}, trying ${FALLBACK_URL}`)
      return fetch(`${FALLBACK_URL}${endpoint}`, options)
    }
    throw error
  }
}
