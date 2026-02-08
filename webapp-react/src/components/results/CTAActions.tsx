import { Button } from '@/components/ui/button'
import { MapPin, Copy, Upload, Share2 } from 'lucide-react'
import { toast } from 'sonner'
import { formatResultsAsText } from '@/lib/downloadUtils'
import type { AnalysisResult } from '@/types'

interface CTAActionsProps {
  tier: 'low' | 'moderate' | 'high'
  results: AnalysisResult
  onAnalyzeAnother?: () => void
}

const FIND_A_DERM_URL = 'https://find-a-derm.aad.org/search?searchTerm=&searchLocation='

const DERM_ACTION = { primary: 'Find a Dermatologist', url: FIND_A_DERM_URL }

const tierActions = {
  low: { primary: 'Learn More', url: 'https://www.aad.org/public/diseases/skin-cancer' },
  moderate: DERM_ACTION,
  high: DERM_ACTION,
}

async function getLocationZipCode(): Promise<string | null> {
  try {
    const position = await new Promise<GeolocationPosition>((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 })
    })
    const { latitude, longitude } = position.coords
    const response = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&zoom=18&addressdetails=1`,
      { headers: { 'User-Agent': 'SkinTag-App' } }
    )
    const data = await response.json()
    return data.address?.postcode || null
  } catch {
    return null
  }
}

export function CTAActions({ tier, results, onAnalyzeAnother }: CTAActionsProps) {
  const actions = tierActions[tier]

  const handleClick = async (url: string) => {
    let targetUrl = url

    if (url.includes('find-a-derm.aad.org')) {
      const zipCode = await getLocationZipCode()
      if (zipCode) {
        targetUrl = url.replace('searchLocation=', `searchLocation=${zipCode}`)
      }
    }

    window.open(targetUrl, '_blank', 'noopener,noreferrer')
  }

  const handleCopy = () => {
    try {
      const text = formatResultsAsText(results)
      navigator.clipboard.writeText(text)
      toast.success('Results copied to clipboard')
    } catch {
      toast.error('Failed to copy to clipboard')
    }
  }

  const handleShare = () => {
    const text = formatResultsAsText(results)
    const shareData = { title: 'SkinTag Analysis Results', text }

    if (navigator.share && navigator.canShare(shareData)) {
      navigator.share(shareData)
        .then(() => toast.success('Results shared'))
        .catch(() => {
          navigator.clipboard.writeText(text)
          toast.success('Results copied to clipboard')
        })
    } else {
      navigator.clipboard.writeText(text)
      toast.success('Results copied - ready to share with doctor')
    }
  }

  return (
    <div className="space-y-3">
      <Button
        onClick={() => handleClick(actions.url)}
        className="w-full"
        size="lg"
      >
        <MapPin className="w-5 h-5" />
        {actions.primary}
      </Button>

      <div className="grid grid-cols-2 gap-3">
        <Button
          onClick={handleShare}
          variant="outline"
          size="default"
        >
          <Share2 className="w-4 h-4" />
          Share
        </Button>
        <Button
          onClick={handleCopy}
          variant="outline"
          size="default"
        >
          <Copy className="w-4 h-4" />
          Copy
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
