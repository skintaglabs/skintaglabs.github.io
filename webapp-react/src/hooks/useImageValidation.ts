import { toast } from 'sonner'
import { useImageQuality } from './useImageQuality'

export function useImageValidation() {
  const { analyzeQuality } = useImageQuality()

  const validateImage = async (file: File): Promise<boolean> => {
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file')
      return false
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('Image must be smaller than 10MB')
      return false
    }

    if (file.size > 5 * 1024 * 1024) {
      toast.warning('Large image detected. Analysis may take longer.')
    }

    try {
      const image = new Image()
      const url = URL.createObjectURL(file)

      await new Promise<void>((resolve, reject) => {
        image.onload = () => {
          URL.revokeObjectURL(url)

          if (image.width > 4000 || image.height > 4000) {
            toast.warning('Very large image. Consider resizing for faster analysis.')
          }

          resolve()
        }
        image.onerror = () => {
          URL.revokeObjectURL(url)
          reject(new Error('Corrupted image'))
        }
        image.src = url
      })

      // Analyze image quality
      try {
        const toastId = toast.loading('Checking image quality...')
        const quality = await analyzeQuality(file)
        toast.dismiss(toastId)

        if (!quality.passed) {
          const primaryIssue = quality.issues[0] || 'Image quality is insufficient'
          toast.error(primaryIssue, {
            description: quality.issues.length > 1
              ? `Also: ${quality.issues.slice(1).join(', ')}`
              : 'Try retaking with better lighting and focus'
          })
          return false
        }

        if (quality.score < 70) {
          toast.warning('Image quality could be better', {
            description: quality.issues.join(', ')
          })
        } else if (quality.score >= 85) {
          toast.success('Image quality looks good!')
        }
      } catch (error) {
        console.warn('Quality analysis failed, proceeding without validation:', error)
      }

      return true
    } catch {
      toast.error('Unable to load image. File may be corrupted.')
      return false
    }
  }

  return { validateImage }
}
