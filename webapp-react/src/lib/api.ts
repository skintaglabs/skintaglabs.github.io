import type { AnalysisResult } from '@/types'
import { FALLBACK_URL } from '@/config/api'

const WARMING_UP = 'WARMING_UP'
const RETRY_DELAY_MS = 15_000
const MAX_RETRIES = 5       // 5 × 15s = 75s max wait for cold start
const TIMEOUT_MS = 90_000   // HF cold start can take ~60-90s

export async function analyzeImage(
  file: File,
  apiUrl: string,
  onWarmingUp?: () => void,
): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)

  const tryAnalyze = async (url: string) => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS)

    try {
      const response = await fetch(`${url}/api/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      clearTimeout(timeoutId)

      // HF Spaces returns 503 while the container is waking from sleep
      if (response.status === 503) throw new Error(WARMING_UP)

      if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || 'Analysis failed')
      }

      return await response.json() as AnalysisResult
    } catch (error) {
      clearTimeout(timeoutId)
      if (error instanceof Error && error.name === 'AbortError') throw new Error('REQUEST_TIMEOUT')
      throw error
    }
  }

  // Retry on 503 (cold start) up to MAX_RETRIES times
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await tryAnalyze(apiUrl)
    } catch (error) {
      const isWarmingUp = error instanceof Error && error.message === WARMING_UP
      const isNetworkError = error instanceof Error && error.message !== 'REQUEST_TIMEOUT' &&
        !error.message.includes('Analysis failed')

      if ((isWarmingUp || isNetworkError) && attempt < MAX_RETRIES) {
        if (attempt === 0) onWarmingUp?.()
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS))
        continue
      }

      // Fall back to localhost on non-warming errors
      if (apiUrl !== FALLBACK_URL) {
        console.warn(`API failed at ${apiUrl}, trying ${FALLBACK_URL}`)
        return await tryAnalyze(FALLBACK_URL)
      }

      throw error
    }
  }

  throw new Error('Server is taking too long to wake up. Please try again in a moment.')
}
