import { useState, useCallback, useEffect } from 'react'
import Cropper from 'react-easy-crop'
import { Button } from '@/components/ui/button'
import { X, Check } from 'lucide-react'
import type { Area } from 'react-easy-crop'
import { useIsMobile } from '@/hooks/useIsMobile'

interface ImageCropperProps {
  imageUrl: string
  onCropComplete: (croppedImageBlob: Blob, croppedImageUrl: string) => void
  onCancel: () => void
}

async function getCroppedImg(imageSrc: string, pixelCrop: Area): Promise<Blob> {
  const image = await createImage(imageSrc)
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  if (!ctx) {
    throw new Error('Canvas context not available')
  }

  canvas.width = pixelCrop.width
  canvas.height = pixelCrop.height

  ctx.drawImage(
    image,
    pixelCrop.x,
    pixelCrop.y,
    pixelCrop.width,
    pixelCrop.height,
    0,
    0,
    pixelCrop.width,
    pixelCrop.height
  )

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
    }, 'image/jpeg', 0.95)
  })
}

function createImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.addEventListener('load', () => resolve(image))
    image.addEventListener('error', (error) => reject(error))
    image.src = url
  })
}

export function ImageCropper({ imageUrl, onCropComplete, onCancel }: ImageCropperProps) {
  const [crop, setCrop] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area | null>(null)
  const isMobile = useIsMobile()

  const handleEscape = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault()
      onCancel()
    }
  }, [onCancel])

  useEffect(() => {
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [handleEscape])

  const onCropCompleteInternal = useCallback((_: Area, croppedAreaPixels: Area) => {
    setCroppedAreaPixels(croppedAreaPixels)
  }, [])

  const handleApplyCrop = async () => {
    if (!croppedAreaPixels) return

    try {
      const croppedBlob = await getCroppedImg(imageUrl, croppedAreaPixels)
      const croppedUrl = URL.createObjectURL(croppedBlob)
      onCropComplete(croppedBlob, croppedUrl)
    } catch (e) {
      console.error('Error cropping image:', e)
    }
  }

  const content = (
    <div className="flex flex-col">
      <div className="relative w-full h-[400px] bg-[var(--color-text)]">
        <Cropper
          image={imageUrl}
          crop={crop}
          zoom={zoom}
          aspect={1}
          onCropChange={setCrop}
          onZoomChange={setZoom}
          onCropComplete={onCropCompleteInternal}
        />
      </div>

      <div className="p-6 space-y-4">
        <div>
          <label className="text-[13px] text-[var(--color-text-muted)] mb-2 block">
            Zoom
          </label>
          <input
            type="range"
            min={1}
            max={3}
            step={0.1}
            value={zoom}
            onChange={(e) => setZoom(Number(e.target.value))}
            className="w-full h-2 bg-[var(--color-surface-alt)] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--color-accent-warm)] [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>

        <div className="text-[13px] text-[var(--color-text-muted)] text-center">
          Drag to reposition • Pinch or use slider to zoom
        </div>

        <div className="flex gap-3">
          <Button onClick={onCancel} variant="outline" className="flex-1">
            <X className="w-4 h-4" />
            Cancel
          </Button>
          <Button onClick={handleApplyCrop} className="flex-1">
            <Check className="w-4 h-4" />
            Apply Crop
          </Button>
        </div>
      </div>
    </div>
  )

  return (
    <div className="fixed inset-0 z-[100]">
      <div className="fixed inset-0 bg-black/40" onClick={onCancel} />
      {isMobile ? (
        <div className="fixed inset-0 bg-[var(--color-surface)] flex flex-col z-[101]">
          {content}
        </div>
      ) : (
        <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-[101] w-full max-w-2xl mx-4 pointer-events-none">
          <div className="bg-[var(--color-surface)] rounded-[var(--radius-lg)] overflow-hidden shadow-[var(--shadow-lg)] pointer-events-auto">
            {content}
          </div>
        </div>
      )}
    </div>
  )
}
