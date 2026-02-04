import { useApiHealth } from '@/hooks/useApiHealth'
import { Button } from '@/components/ui/button'

export function ApiHealthBanner() {
  const { isHealthy, checkHealth } = useApiHealth()

  if (isHealthy === null || isHealthy === true) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-amber-500 text-white px-4 py-3 text-center text-sm font-medium shadow-lg">
      <div className="flex items-center justify-center gap-3">
        <span>API endpoint unavailable. The server may have restarted.</span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => window.location.reload()}
          className="bg-white text-amber-900 hover:bg-amber-50 border-amber-200"
        >
          Refresh Page
        </Button>
        <button
          onClick={checkHealth}
          className="text-white underline hover:no-underline"
        >
          Retry
        </button>
      </div>
    </div>
  )
}
