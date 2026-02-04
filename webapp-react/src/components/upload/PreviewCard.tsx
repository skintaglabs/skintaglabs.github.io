import { useState, useEffect } from 'react'
import { X, FileImage, Crop } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { formatFileSize } from '@/lib/utils'

interface PreviewCardProps {
  file: File
  previewUrl: string
  onClear: () => void
  onAnalyze: () => void
  onCrop: () => void
}

export function PreviewCard({ file, previewUrl, onClear, onAnalyze, onCrop }: PreviewCardProps) {
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null)

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      setDimensions({ width: img.width, height: img.height })
    }
    img.src = previewUrl
  }, [previewUrl])

  return (
    <Card className="overflow-hidden">
      <div className="relative aspect-[4/3] bg-[var(--color-surface-alt)]">
        <img
          src={previewUrl}
          alt="Preview"
          className="w-full h-full object-contain"
        />
        <button
          onClick={onClear}
          className="absolute top-3 right-3 w-8 h-8 rounded-full bg-[var(--color-text)]/80 hover:bg-[var(--color-text)] text-[var(--color-surface)] flex items-center justify-center transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
      <div className="p-4 space-y-3">
        <div className="flex items-start gap-3">
          <FileImage className="w-5 h-5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-[15px] font-medium truncate">{file.name}</p>
            <p className="text-[13px] text-[var(--color-text-muted)] mt-0.5">
              {formatFileSize(file.size)}
              {dimensions && ` • ${dimensions.width} × ${dimensions.height}px`}
            </p>
          </div>
        </div>
        <div className="flex gap-3">
          <Button onClick={onCrop} variant="outline" className="flex-1">
            <Crop className="w-4 h-4" />
            Crop
          </Button>
          <Button onClick={onAnalyze} className="flex-1">
            Analyze
          </Button>
        </div>
      </div>
    </Card>
  )
}
