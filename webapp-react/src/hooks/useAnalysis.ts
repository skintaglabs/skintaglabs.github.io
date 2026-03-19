import { useAppContext } from '@/contexts/AppContext'
import { useNetworkStatus } from './useNetworkStatus'
import { analyzeImage } from '@/lib/api'
import { API_URL } from '@/config/api'
import { toast } from 'sonner'
import type { AnalysisResult } from '@/types'

const WARMUP_TOAST_ID = 'server-warmup'

export function useAnalysis() {
  const { setIsAnalyzing, setResults } = useAppContext()
  const { isOnline } = useNetworkStatus()

  const analyze = async (file: File): Promise<AnalysisResult | null> => {
    if (!isOnline) {
      toast.error('No internet connection')
      return null
    }

    setIsAnalyzing(true)

    const slowTimeout = setTimeout(() => {
      toast.info('Still analyzing...', { duration: Infinity })
    }, 10000)

    try {
      const result = await analyzeImage(file, API_URL, () => {
        clearTimeout(slowTimeout)
        toast.loading('Server is warming up, please wait...', {
          id: WARMUP_TOAST_ID,
          duration: Infinity,
        })
      })

      clearTimeout(slowTimeout)
      toast.dismiss(WARMUP_TOAST_ID)
      toast.dismiss()
      setResults(result)
      toast.success('Analysis complete', { duration: 2000 })
      return result
    } catch (error) {
      clearTimeout(slowTimeout)
      toast.dismiss(WARMUP_TOAST_ID)
      toast.dismiss()

      if (error instanceof Error && error.message === 'REQUEST_TIMEOUT') {
        toast.error('Request timed out. Try a smaller image.', {
          action: { label: 'Retry', onClick: () => analyze(file) },
        })
      } else if (error instanceof Error && error.message.includes('wake up')) {
        toast.error('Server took too long to wake up.', {
          description: 'Please try again in a moment.',
          action: { label: 'Retry', onClick: () => analyze(file) },
        })
      } else {
        toast.error('Analysis failed', {
          description: error instanceof Error ? error.message : 'Unexpected error',
          action: { label: 'Retry', onClick: () => analyze(file) },
        })
      }
      return null
    } finally {
      setIsAnalyzing(false)
    }
  }

  return { analyze }
}
