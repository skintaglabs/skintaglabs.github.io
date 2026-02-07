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
    <>
      <div className="fixed inset-0 z-[100] bg-black/40" onClick={onClear} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-[100] w-full max-w-2xl mx-4 max-h-[90vh] flex flex-col animate-fadeUp">
        <div className="bg-[var(--color-surface)] rounded-[var(--radius-lg)] overflow-hidden shadow-[var(--shadow-lg)] flex flex-col max-h-[90vh]">
          <div className="relative bg-[var(--color-surface-alt)] flex items-center justify-center min-h-0 flex-1" style={{ maxHeight: '400px' }}>
            <img
              src={previewUrl}
              alt="Preview"
              className="max-w-full max-h-full object-contain preview-image"
            />
            <button
              onClick={onClear}
              className="absolute top-3 right-3 w-10 h-10 rounded-full bg-[var(--color-text)]/80 hover:bg-[var(--color-text)] text-[var(--color-surface)] flex items-center justify-center transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="p-6 space-y-4 flex-shrink-0">
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
      </div>
    </>
  )
}
