import { useApiHealth } from '@/hooks/useApiHealth'

export function ApiHealthBanner() {
  const { isHealthy, checkHealth, isChecking } = useApiHealth()

  if (isHealthy !== false) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-amber-500 text-white px-4 py-3 text-center text-sm font-medium shadow-lg">
      <div className="flex items-center justify-center gap-3">
        <span>API endpoint unavailable. The server may have restarted.</span>
        <button
          onClick={checkHealth}
          disabled={isChecking}
          className="text-white underline hover:no-underline disabled:opacity-50"
        >
          {isChecking ? 'Checking...' : 'Retry'}
        </button>
      </div>
    </div>
  )
}
