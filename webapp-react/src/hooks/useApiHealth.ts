import { useState, useEffect } from 'react'
import { fetchWithFallback } from '@/config/api'

export function useApiHealth() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)
  const [isChecking, setIsChecking] = useState(false)

  const checkHealth = async () => {
    if (isChecking) return

    setIsChecking(true)
    try {
      const response = await fetchWithFallback('/api/health', {
        cache: 'no-store'
      })

      setIsHealthy(response.ok)
    } catch {
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
