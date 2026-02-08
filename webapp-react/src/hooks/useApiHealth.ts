import { useState, useEffect } from 'react'
import { API_URL, FALLBACK_URL } from '@/config/api'

export function useApiHealth() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)
  const [isChecking, setIsChecking] = useState(false)

  const checkHealth = async () => {
    if (isChecking) return

    setIsChecking(true)
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000)
      const response = await fetch(`${API_URL}/api/health`, {
        cache: 'no-store',
        signal: controller.signal
      })
      clearTimeout(timeoutId)
      setIsHealthy(response.ok)
    } catch {
      if (API_URL !== FALLBACK_URL) {
        try {
          const response = await fetch(`${FALLBACK_URL}/api/health`, { cache: 'no-store' })
          setIsHealthy(response.ok)
          return
        } catch { /* fall through */ }
      }
      setIsHealthy(false)
    } finally {
      setIsChecking(false)
    }
  }

  useEffect(() => {
    checkHealth()

    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  return { isHealthy, checkHealth, isChecking }
}
