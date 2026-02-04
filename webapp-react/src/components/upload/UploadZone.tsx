import { useState, type DragEvent, type ChangeEvent } from 'react'
import { Upload } from 'lucide-react'
import { useImageValidation } from '@/hooks/useImageValidation'

interface UploadZoneProps {
  onFileSelect: (file: File, previewUrl: string) => void
}

export function UploadZone({ onFileSelect }: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const { validateImage } = useImageValidation()

  const handleFile = async (file: File) => {
    const isValid = await validateImage(file)
    if (isValid) {
      const previewUrl = URL.createObjectURL(file)
      onFileSelect(file, previewUrl)
    }
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFile(file)
    }
  }

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    setIsDragOver(false)
  }

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFile(file)
    }
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`
        relative border-2 border-dashed rounded-[var(--radius-lg)] p-12
        transition-all duration-[var(--duration-normal)]
        ${isDragOver
          ? 'border-[var(--color-accent-warm)] bg-[var(--color-surface-alt)] scale-[1.02] drag-over-pulse'
          : 'border-[var(--color-border)] bg-[var(--color-surface)]'
        }
      `}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleInputChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        id="file-upload"
      />
      <div className="flex flex-col items-center gap-4 pointer-events-none">
        <div className={`
          w-16 h-16 rounded-full flex items-center justify-center
          transition-colors duration-[var(--duration-normal)]
          ${isDragOver ? 'bg-[var(--color-accent-warm)]' : 'bg-[var(--color-surface-alt)]'}
        `}>
          <Upload className={`w-8 h-8 ${isDragOver ? 'text-[var(--color-surface)]' : 'text-[var(--color-text-muted)] upload-icon-idle'}`} />
        </div>
        <div className="text-center">
          <p className="text-[17px] font-medium mb-1">Drop your image here or click to browse</p>
          <p className="text-[15px] text-[var(--color-text-muted)]">
            JPG, PNG, or HEIC • Maximum 10MB
          </p>
        </div>
        <div className="text-[13px] text-[var(--color-text-muted)] text-center max-w-md mt-2">
          For best results, use a clear photo of the skin lesion with good lighting.
          Dermoscopic images work best.
        </div>
      </div>
    </div>
  )
}
