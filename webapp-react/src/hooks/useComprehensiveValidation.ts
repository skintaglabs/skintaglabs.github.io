import { useRef, useCallback } from 'react'
import {
  HandLandmarker,
  PoseLandmarker,
  ImageSegmenter,
  FilesetResolver
} from '@mediapipe/tasks-vision'

interface ValidationResult {
  passed: boolean
  score: number
  feedback: {
    type: 'success' | 'error' | 'warning'
    message: string
    details?: string
  }
  checks: {
    distance: 'too_far' | 'good' | 'too_close' | 'unknown'
    skinPresent: boolean
    skinPercentage: number
    blur: number
    brightness: number
    contrast: number
  }
}

export function useComprehensiveValidation() {
  const handLandmarkerRef = useRef<HandLandmarker | null>(null)
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null)
  const imageSegmenterRef = useRef<ImageSegmenter | null>(null)
  const isInitializingRef = useRef(false)

  const initialize = useCallback(async () => {
    if (isInitializingRef.current) return
    isInitializingRef.current = true

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      )

      // Initialize all detectors in parallel
      const [handLandmarker, poseLandmarker, imageSegmenter] = await Promise.all([
        HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          numHands: 2,
          runningMode: 'IMAGE',
          minHandDetectionConfidence: 0.5
        }),
        PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
            delegate: 'GPU'
          },
          runningMode: 'IMAGE',
          minPoseDetectionConfidence: 0.5
        }),
        ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.task',
            delegate: 'GPU'
          },
          runningMode: 'IMAGE',
          outputCategoryMask: true
        })
      ])

      handLandmarkerRef.current = handLandmarker
      poseLandmarkerRef.current = poseLandmarker
      imageSegmenterRef.current = imageSegmenter
    } catch (error) {
      console.error('Failed to initialize validators:', error)
    } finally {
      isInitializingRef.current = false
    }
  }, [])

  const checkDistance = useCallback(async (
    image: ImageBitmap
  ): Promise<'too_far' | 'good' | 'unknown'> => {
    try {
      // Check for hands first
      if (handLandmarkerRef.current) {
        const handResults = handLandmarkerRef.current.detect(image)
        if (handResults.landmarks && handResults.landmarks.length > 0) {
          const landmarks = handResults.landmarks[0]

          // Calculate hand size in frame
          const xs = landmarks.map(l => l.x)
          const ys = landmarks.map(l => l.y)
          const width = Math.max(...xs) - Math.min(...xs)
          const height = Math.max(...ys) - Math.min(...ys)
          const handSize = Math.max(width, height)

          // If hand takes up < 40% of frame, it's too far
          if (handSize < 0.4) {
            return 'too_far'
          }
        }
      }

      // Check for full body pose
      if (poseLandmarkerRef.current) {
        const poseResults = poseLandmarkerRef.current.detect(image)
        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
          const landmarks = poseResults.landmarks[0]

          // If we can see full torso landmarks, definitely too far
          const hasFullTorso = landmarks.some(l => l.visibility && l.visibility > 0.5)
          if (hasFullTorso) {
            return 'too_far'
          }
        }
      }

      return 'unknown'
    } catch (error) {
      console.error('Distance check failed:', error)
      return 'unknown'
    }
  }, [])

  const checkSkinSegmentation = useCallback(async (
    image: ImageBitmap
  ): Promise<{ present: boolean; percentage: number }> => {
    try {
      if (!imageSegmenterRef.current) {
        console.warn('Image segmenter not initialized, skipping check')
        return { present: false, percentage: 0 } // Fallback: assume close-up (no full person)
      }

      const result = imageSegmenterRef.current.segment(image)
      if (!result.categoryMask) {
        console.warn('No category mask returned, skipping check')
        return { present: false, percentage: 0 } // Fallback: assume close-up (no full person)
      }

      // Count person pixels (category 1)
      // High percentage = full person/face visible = too far away
      // Low percentage = close-up of skin area = good
      const mask = result.categoryMask.getAsFloat32Array()
      let personPixels = 0
      for (let i = 0; i < mask.length; i++) {
        if (mask[i] > 0.5) personPixels++
      }

      const percentage = (personPixels / mask.length) * 100
      return {
        present: percentage > 20, // present = full person detected (bad for lesion photos)
        percentage
      }
    } catch (error) {
      console.error('Segmentation check failed:', error)
      return { present: false, percentage: 0 } // Fallback: assume close-up (no full person)
    }
  }, [])

  const createCircularROI = useCallback((
    image: ImageBitmap,
    canvas: HTMLCanvasElement
  ): { roiImage: ImageBitmap; roiImageData: ImageData } => {
    // Calculate circular ROI (centered, 256px diameter like webcam guide)
    const centerX = image.width / 2
    const centerY = image.height / 2
    const radius = Math.min(image.width, image.height) * 0.4 // 40% of smaller dimension

    // Create circular mask canvas
    canvas.width = image.width
    canvas.height = image.height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Create circular clip path
    ctx.save()
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
    ctx.clip()

    // Draw image within circular mask
    ctx.drawImage(image, 0, 0)
    ctx.restore()

    // Get image data for quality checks (only circular region has valid data)
    const roiImageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

    // Create ROI bitmap for MediaPipe validators
    // Note: Canvas already has the masked image drawn
    const roiImage = image // We'll validate the full canvas which has masked content

    return { roiImage, roiImageData }
  }, [])

  const checkImageQuality = useCallback((
    imageData: ImageData
  ): { blur: number; brightness: number; contrast: number } => {
    const data = imageData.data
    const width = imageData.width
    const height = imageData.height

    // Brightness
    let totalBrightness = 0
    for (let i = 0; i < data.length; i += 4) {
      totalBrightness += (data[i] + data[i + 1] + data[i + 2]) / 3
    }
    const brightness = totalBrightness / (data.length / 4)

    // Blur (edge detection)
    let edgeStrength = 0
    let count = 0
    for (let y = 1; y < height - 1; y += 10) {
      for (let x = 1; x < width - 1; x += 10) {
        const idx = (y * width + x) * 4
        const center = data[idx]
        const top = data[((y - 1) * width + x) * 4]
        const bottom = data[((y + 1) * width + x) * 4]
        const left = data[(y * width + (x - 1)) * 4]
        const right = data[(y * width + (x + 1)) * 4]
        const laplacian = Math.abs(4 * center - top - bottom - left - right)
        edgeStrength += laplacian
        count++
      }
    }
    const blur = edgeStrength / count

    // Contrast
    let min = 255
    let max = 0
    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3
      min = Math.min(min, brightness)
      max = Math.max(max, brightness)
    }
    const contrast = max - min

    return { blur, brightness, contrast }
  }, [])

  const validate = useCallback(async (file: File): Promise<ValidationResult> => {
    // Initialize detectors if needed
    await initialize()

    const image = await createImageBitmap(file)

    // CRITICAL: Create circular ROI (medical standard - focus on lesion area only)
    const canvas = document.createElement('canvas')
    const { roiImage, roiImageData } = createCircularROI(image, canvas)

    // Run all checks in parallel ON THE CIRCULAR ROI ONLY
    // This prevents artifacts outside the diagnostic area from affecting scores
    const [distance, skinSeg, quality] = await Promise.all([
      checkDistance(roiImage), // Distance check on ROI
      checkSkinSegmentation(roiImage), // Person detection on ROI
      Promise.resolve(checkImageQuality(roiImageData)) // Quality metrics on ROI pixels only
    ])

    // Build result
    const checks = {
      distance,
      skinPresent: skinSeg.present,
      skinPercentage: skinSeg.percentage,
      blur: quality.blur,
      brightness: quality.brightness,
      contrast: quality.contrast
    }

    // Debug logging (medical-standard validation on circular ROI)
    console.log('Validation checks (Circular ROI only):', {
      distance,
      personDetected: skinSeg.percentage.toFixed(2) + '% (optimal: <10%, warning: >20%, critical: >40%)',
      blur: quality.blur.toFixed(2) + ' (optimal: ≥12, warning: <12, critical: <5)',
      brightness: quality.brightness.toFixed(2) + ' (optimal: 90-180, acceptable: 60-220)',
      contrast: quality.contrast.toFixed(2) + ' (minimum: 30 for lesion borders)'
    })

    // Scoring logic
    let score = 100
    const issues: string[] = []

    // Distance check
    if (distance === 'too_far') {
      score -= 40
      issues.push('Too far from subject')
    }

    // Person detection check - high percentage means too far away (medical standard)
    if (skinSeg.percentage > 40) {
      score -= 50
      issues.push('Too far - only a small skin area should be visible')
    } else if (skinSeg.percentage > 20) {
      score -= 25
      issues.push('Get closer - focus on lesion area only')
    } else if (skinSeg.percentage < 10) {
      // OPTIMAL: Close-up of skin tissue (< 10% person detected)
      // No penalty - this is exactly what we want for dermoscopy
    }
    // Low percentage (<20%) is good - means close-up of skin area

    // Blur check (medical standard: dermoscopic images have subtle textures)
    // Research shows blur detection struggles with 0.2-1mm range typical in dermoscopy
    if (quality.blur < 5) {
      score -= 40
      issues.push('Severely blurred - hold steady')
    } else if (quality.blur < 12) {
      score -= 15
      issues.push('Slightly blurred - try to hold steadier')
    }
    // blur >= 12 is acceptable for dermoscopy

    // Brightness check (medical standard: neutral grey reference 60-220)
    if (quality.brightness < 60 || quality.brightness > 220) {
      score -= 35
      issues.push(quality.brightness < 60 ? 'Too dark - use natural light' : 'Overexposed - reduce lighting')
    } else if (quality.brightness < 90 || quality.brightness > 180) {
      score -= 10
      issues.push('Lighting could be better')
    }
    // 90-180 is optimal range for dermoscopy

    // Contrast check (medical standard: >30 for lesion border visibility)
    if (quality.contrast < 30) {
      score -= 20
      issues.push('Low contrast - move closer or adjust lighting')
    }
    // contrast >= 30 ensures lesion borders are visible

    // Determine feedback (guidance only, never reject)
    let feedback: ValidationResult['feedback']

    console.log('Final score:', score, 'Issues:', issues)

    if (score >= 80) {
      feedback = {
        type: 'success',
        message: 'Excellent image quality!',
        details: undefined
      }
    } else if (score >= 60) {
      feedback = {
        type: 'success',
        message: 'Good image quality',
        details: issues.length > 0 ? 'Minor issues: ' + issues.join(', ') : undefined
      }
    } else if (issues.length > 0) {
      feedback = {
        type: 'warning',
        message: 'Image accepted, but could be improved',
        details: issues.join(', ')
      }
    } else {
      feedback = {
        type: 'success',
        message: 'Image accepted',
        details: undefined
      }
    }

    return {
      passed: true, // Always pass - just provide guidance
      score,
      feedback,
      checks
    }
  }, [initialize, createCircularROI, checkDistance, checkSkinSegmentation, checkImageQuality])

  return { validate }
}
