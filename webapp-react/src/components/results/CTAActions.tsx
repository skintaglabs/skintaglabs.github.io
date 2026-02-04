import { Button } from '@/components/ui/button'
import { MapPin, Download, Copy, Upload } from 'lucide-react'
import { toast } from 'sonner'
import { downloadResultsAsImage, formatResultsAsText, copyResultsToClipboard } from '@/lib/downloadUtils'
import type { AnalysisResult } from '@/types'

interface CTAActionsProps {
  tier: 'low' | 'moderate' | 'high'
  results: AnalysisResult
  onAnalyzeAnother?: () => void
}

const tierActions = {
  low: {
    primary: 'Track Over Time'
  },
  moderate: {
    primary: 'Schedule Consultation'
  },
  high: {
    primary: 'Book Dermatologist Visit'
  }
}

export function CTAActions({ tier, results, onAnalyzeAnother }: CTAActionsProps) {
  const actions = tierActions[tier]

  const handleClick = (action: string) => {
    toast.info('Feature coming soon!', {
      description: `${action} will be available in a future update.`
    })
  }

  const handleDownload = async () => {
    try {
      await downloadResultsAsImage('results-capture')
      toast.success('Results saved as image')
    } catch (error) {
      console.error('Download error:', error)
      toast.error('Failed to save image', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleCopy = () => {
    try {
      const text = formatResultsAsText(results)
      copyResultsToClipboard(text)
      toast.success('Results copied to clipboard')
    } catch (error) {
      toast.error('Failed to copy to clipboard')
    }
  }

  return (
    <div className="space-y-3">
      <Button
        onClick={() => handleClick(actions.primary)}
        className="w-full"
        size="lg"
      >
        <MapPin className="w-5 h-5" />
        {actions.primary}
      </Button>

      <div className="grid grid-cols-2 gap-3">
        <Button
          onClick={handleDownload}
          variant="outline"
          size="default"
        >
          <Download className="w-4 h-4" />
          Save Image
        </Button>
        <Button
          onClick={handleCopy}
          variant="outline"
          size="default"
        >
          <Copy className="w-4 h-4" />
          Copy Text
        </Button>
      </div>

      {onAnalyzeAnother && (
        <Button
          onClick={onAnalyzeAnother}
          variant="ghost"
          className="w-full"
        >
          <Upload className="w-4 h-4" />
          Analyze Another Image
        </Button>
      )}
    </div>
  )
}
