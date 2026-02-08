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
      if (handLandmarkerRef.current) {
        const handResults = handLandmarkerRef.current.detect(image)
        if (handResults.landmarks && handResults.landmarks.length > 0) {
          const landmarks = handResults.landmarks[0]

          const xs = landmarks.map(l => l.x)
          const ys = landmarks.map(l => l.y)
          const width = Math.max(...xs) - Math.min(...xs)
          const height = Math.max(...ys) - Math.min(...ys)
          const handSize = Math.max(width, height)

          if (handSize < 0.4) {
            return 'too_far'
          }
        }
      }

      if (poseLandmarkerRef.current) {
        const poseResults = poseLandmarkerRef.current.detect(image)
        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
          const landmarks = poseResults.landmarks[0]

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
        return { present: false, percentage: 0 }
      }

      const result = imageSegmenterRef.current.segment(image)
      if (!result.categoryMask) {
        return { present: false, percentage: 0 }
      }

      const mask = result.categoryMask.getAsFloat32Array()
      let personPixels = 0
      for (let i = 0; i < mask.length; i++) {
        if (mask[i] > 0.5) personPixels++
      }

      const percentage = (personPixels / mask.length) * 100
      return {
        present: percentage > 20,
        percentage
      }
    } catch (error) {
      console.error('Segmentation check failed:', error)
      return { present: false, percentage: 0 }
    }
  }, [])

  const createCircularROI = useCallback((
    image: ImageBitmap,
    canvas: HTMLCanvasElement
  ): { roiImage: ImageBitmap; roiImageData: ImageData } => {
    const centerX = image.width / 2
    const centerY = image.height / 2
    const radius = Math.min(image.width, image.height) * 0.4

    canvas.width = image.width
    canvas.height = image.height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.save()
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
    ctx.clip()
    ctx.drawImage(image, 0, 0)
    ctx.restore()

    const roiImageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

    // MediaPipe detectors receive the full image (circular masking only affects quality metrics)
    return { roiImage: image, roiImageData }
  }, [])

  const checkImageQuality = useCallback((
    imageData: ImageData
  ): { blur: number; brightness: number; contrast: number } => {
    const data = imageData.data
    const width = imageData.width
    const height = imageData.height

    let totalBrightness = 0
    for (let i = 0; i < data.length; i += 4) {
      totalBrightness += (data[i] + data[i + 1] + data[i + 2]) / 3
    }
    const brightness = totalBrightness / (data.length / 4)

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
    await initialize()

    const image = await createImageBitmap(file)

    const canvas = document.createElement('canvas')
    const { roiImage, roiImageData } = createCircularROI(image, canvas)

    const [distance, skinSeg, quality] = await Promise.all([
      checkDistance(roiImage),
      checkSkinSegmentation(roiImage),
      Promise.resolve(checkImageQuality(roiImageData))
    ])

    const checks = {
      distance,
      skinPresent: skinSeg.present,
      skinPercentage: skinSeg.percentage,
      blur: quality.blur,
      brightness: quality.brightness,
      contrast: quality.contrast
    }

    let score = 100
    const issues: string[] = []

    if (distance === 'too_far') {
      score -= 40
      issues.push('Too far from subject')
    }

    if (skinSeg.percentage > 40) {
      score -= 50
      issues.push('Too far - only a small skin area should be visible')
    } else if (skinSeg.percentage > 20) {
      score -= 25
      issues.push('Get closer - focus on lesion area only')
    }

    if (quality.blur < 5) {
      score -= 40
      issues.push('Severely blurred - hold steady')
    } else if (quality.blur < 12) {
      score -= 15
      issues.push('Slightly blurred - try to hold steadier')
    }

    if (quality.brightness < 60 || quality.brightness > 220) {
      score -= 35
      issues.push(quality.brightness < 60 ? 'Too dark - use natural light' : 'Overexposed - reduce lighting')
    } else if (quality.brightness < 90 || quality.brightness > 180) {
      score -= 10
      issues.push('Lighting could be better')
    }

    if (quality.contrast < 30) {
      score -= 20
      issues.push('Low contrast - move closer or adjust lighting')
    }

    let feedback: ValidationResult['feedback']

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
    } else {
      feedback = {
        type: 'warning',
        message: 'Image accepted, but could be improved',
        details: issues.join(', ')
      }
    }

    return {
      passed: true,
      score,
      feedback,
      checks
    }
  }, [initialize, createCircularROI, checkDistance, checkSkinSegmentation, checkImageQuality])

  return { validate }
}
