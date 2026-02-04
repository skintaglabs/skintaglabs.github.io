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
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/2/selfie_segmenter.task',
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
        return { present: true, percentage: 50 } // Graceful fallback
      }

      const result = imageSegmenterRef.current.segment(image)
      if (!result.categoryMask) {
        console.warn('No category mask returned, skipping check')
        return { present: true, percentage: 50 } // Graceful fallback
      }

      // Count person pixels (category 1)
      const mask = result.categoryMask.getAsFloat32Array()
      let personPixels = 0
      for (let i = 0; i < mask.length; i++) {
        if (mask[i] > 0.5) personPixels++
      }

      const percentage = (personPixels / mask.length) * 100
      return {
        present: percentage > 10,
        percentage
      }
    } catch (error) {
      console.error('Segmentation check failed:', error)
      return { present: true, percentage: 50 } // Graceful fallback
    }
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

    // Create canvas for quality checks
    const canvas = document.createElement('canvas')
    canvas.width = image.width
    canvas.height = image.height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(image, 0, 0)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

    // Run all checks in parallel
    const [distance, skinSeg, quality] = await Promise.all([
      checkDistance(image),
      checkSkinSegmentation(image),
      Promise.resolve(checkImageQuality(imageData))
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

    // Debug logging
    console.log('Validation checks:', {
      distance,
      skinPercentage: skinSeg.percentage.toFixed(2) + '%',
      blur: quality.blur.toFixed(2),
      brightness: quality.brightness.toFixed(2),
      contrast: quality.contrast.toFixed(2)
    })

    // Scoring logic
    let score = 100
    const issues: string[] = []

    // Distance check
    if (distance === 'too_far') {
      score -= 40
      issues.push('Too far from subject')
    }

    // Skin presence check (relaxed - only fail if 0%)
    if (skinSeg.percentage === 0) {
      score -= 40
      issues.push('No skin detected')
    } else if (skinSeg.percentage < 5) {
      score -= 20
      issues.push('Not enough skin visible')
    }

    // Blur check (relaxed thresholds)
    if (quality.blur < 20) {
      score -= 30
      issues.push('Image is blurry')
    } else if (quality.blur < 40) {
      score -= 10
      issues.push('Could be sharper')
    }

    // Brightness check
    if (quality.brightness < 40 || quality.brightness > 240) {
      score -= 30
      issues.push(quality.brightness < 40 ? 'Too dark' : 'Overexposed')
    } else if (quality.brightness < 80) {
      score -= 15
      issues.push('Low lighting')
    }

    // Contrast check
    if (quality.contrast < 20) {
      score -= 15
      issues.push('Low contrast')
    }

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
  }, [initialize, checkDistance, checkSkinSegmentation, checkImageQuality])

  return { validate }
}
