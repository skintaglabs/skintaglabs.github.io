import { useState, useEffect } from 'react'
import { X, FileImage, Crop } from 'lucide-react'
import { Button } from '@/components/ui/button'
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
    <div className="fixed inset-0 z-[100] bg-[var(--color-bg)] flex flex-col">
      <div className="flex-1 flex items-center justify-center bg-[var(--color-surface-alt)] relative">
        <img
          src={previewUrl}
          alt="Preview"
          className="max-w-full max-h-full object-contain"
        />
        <button
          onClick={onClear}
          className="absolute top-4 right-4 w-10 h-10 rounded-full bg-[var(--color-text)]/80 hover:bg-[var(--color-text)] text-[var(--color-surface)] flex items-center justify-center transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      <div className="bg-[var(--color-surface)] border-t border-[var(--color-border)] p-6 space-y-4 safe-area-pb">
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
    </div>
  )
}
