import { toast } from 'sonner'
import { useComprehensiveValidation } from './useComprehensiveValidation'

export function useImageValidation() {
  const { validate: comprehensiveValidate } = useComprehensiveValidation()

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

      try {
        const toastId = toast.loading('Analyzing image...')
        const result = await comprehensiveValidate(file)
        toast.dismiss(toastId)

        if (!result.passed) {
          toast.error(result.feedback.message, {
            description: result.feedback.details
          })
          return false
        }

        if (result.feedback.type === 'warning') {
          toast.warning(result.feedback.message, {
            description: result.feedback.details
          })
        } else if (result.feedback.type === 'success') {
          toast.success(result.feedback.message, {
            description: result.feedback.details
          })
        }
      } catch (error) {
        console.warn('Comprehensive validation failed, proceeding without validation:', error)
      }

      return true
    } catch {
      toast.error('Unable to load image. File may be corrupted.')
      return false
    }
  }

  return { validateImage }
}
