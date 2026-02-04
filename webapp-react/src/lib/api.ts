import type { AnalysisResult } from '@/types'
import { FALLBACK_URL } from '@/config/api'

export async function analyzeImage(file: File, apiUrl: string): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)

  const tryAnalyze = async (url: string) => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 60000)

    try {
      const response = await fetch(`${url}/api/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Analysis failed')
      }

      return await response.json()
    } catch (error) {
      clearTimeout(timeoutId)
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('REQUEST_TIMEOUT')
      }
      throw error
    }
  }

  // Try configured URL first
  try {
    return await tryAnalyze(apiUrl)
  } catch (error) {
    // If configured URL fails and it's not localhost, try localhost
    if (apiUrl !== FALLBACK_URL) {
      console.warn(`API failed at ${apiUrl}, trying ${FALLBACK_URL}`)
      return await tryAnalyze(FALLBACK_URL)
    }
    throw error
  }
}
