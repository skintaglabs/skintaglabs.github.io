import type { AnalysisResult } from '@/types'
import { FALLBACK_URL } from '@/config/api'

const MAX_DIMENSION = 1024

function resizeImage(file: File): Promise<File> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const { width, height } = img
      if (width <= MAX_DIMENSION && height <= MAX_DIMENSION) {
        URL.revokeObjectURL(img.src)
        resolve(file)
        return
      }

      const scale = MAX_DIMENSION / Math.max(width, height)
      const canvas = document.createElement('canvas')
      canvas.width = Math.round(width * scale)
      canvas.height = Math.round(height * scale)
      canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(img.src)

      canvas.toBlob(
        (blob) => resolve(new File([blob!], file.name, { type: 'image/jpeg' })),
        'image/jpeg',
        0.90
      )
    }
    img.src = URL.createObjectURL(file)
  })
}

export async function analyzeImage(file: File, apiUrl: string): Promise<AnalysisResult> {
  const resized = await resizeImage(file)
  const formData = new FormData()
  formData.append('file', resized)

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
