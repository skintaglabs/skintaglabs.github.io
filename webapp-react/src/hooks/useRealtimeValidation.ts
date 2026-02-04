import { useRef, useCallback, useEffect } from 'react'
import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
  type NormalizedLandmark
} from '@mediapipe/tasks-vision'

interface RealtimeMetrics {
  distance: 'too_far' | 'good' | 'too_close' | 'unknown'
  handDetected: boolean
  blur: number
  brightness: number
  readyToCapture: boolean
  suggestions: string[]
  landmarks: NormalizedLandmark[][]
}

export function useRealtimeValidation() {
  const handLandmarkerRef = useRef<HandLandmarker | null>(null)
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null)
  const isInitializedRef = useRef(false)

  const initialize = useCallback(async () => {
    if (isInitializedRef.current) return
    isInitializedRef.current = true

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      )

      const [handLandmarker, poseLandmarker] = await Promise.all([
        HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          numHands: 2,
          runningMode: 'VIDEO',
          minHandDetectionConfidence: 0.3
        }),
        PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          minPoseDetectionConfidence: 0.3
        })
      ])

      handLandmarkerRef.current = handLandmarker
      poseLandmarkerRef.current = poseLandmarker
    } catch (error) {
      console.error('Failed to initialize realtime validators:', error)
    }
  }, [])

  const analyzeFrame = useCallback((
    video: HTMLVideoElement,
    timestamp: number
  ): RealtimeMetrics => {
    const suggestions: string[] = []
    let distance: RealtimeMetrics['distance'] = 'unknown'
    let handDetected = false
    let landmarks: NormalizedLandmark[][] = []

    // Check for hands
    if (handLandmarkerRef.current && video.readyState >= 2) {
      try {
        const handResults = handLandmarkerRef.current.detectForVideo(video, timestamp)
        if (handResults.landmarks && handResults.landmarks.length > 0) {
          handDetected = true
          landmarks = handResults.landmarks
          const handLandmarks = handResults.landmarks[0]

          // Calculate hand size
          const xs = handLandmarks.map(l => l.x)
          const ys = handLandmarks.map(l => l.y)
          const width = Math.max(...xs) - Math.min(...xs)
          const height = Math.max(...ys) - Math.min(...ys)
          const handSize = Math.max(width, height)

          // Check if it's a full hand or partial view
          // Full hand = all key landmarks visible (fingertips)
          const thumbTip = handLandmarks[4]
          const indexTip = handLandmarks[8]
          const middleTip = handLandmarks[12]
          const ringTip = handLandmarks[16]
          const pinkyTip = handLandmarks[20]

          // If we can see wrist + multiple fingertips clearly, it's likely a full hand
          const fingertipsVisible = [thumbTip, indexTip, middleTip, ringTip, pinkyTip]
            .filter(tip => tip.x > 0 && tip.x < 1 && tip.y > 0 && tip.y < 1)
          const isFullHand = fingertipsVisible.length >= 4 && handSize < 0.6

          if (isFullHand) {
            distance = 'too_far'
            suggestions.push('Full hand visible - zoom in on the specific area')
          } else if (handSize < 0.3) {
            distance = 'too_far'
            suggestions.push('Move closer')
          } else if (handSize > 0.8) {
            distance = 'too_close'
            suggestions.push('Move back slightly')
          } else {
            distance = 'good'
          }
        } else {
          // No hand detected - might be facial lesion or other body part
          // Don't penalize, just inform
          suggestions.push('Position the lesion in the circle')
        }
      } catch (error) {
        console.warn('Hand detection frame failed:', error)
      }
    }

    // Pose detection removed - caused false negatives for facial lesions
    // For facial/neck lesions, showing the face is unavoidable and correct
    // Focus on practical checks: hand distance, blur, brightness

    // Quick blur estimate
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(video, 0, 0)

    const imageData = ctx.getImageData(
      canvas.width / 2 - 50,
      canvas.height / 2 - 50,
      100,
      100
    )
    const data = imageData.data

    // Calculate brightness
    let totalBrightness = 0
    for (let i = 0; i < data.length; i += 4) {
      totalBrightness += (data[i] + data[i + 1] + data[i + 2]) / 3
    }
    const brightness = totalBrightness / (data.length / 4)

    if (brightness < 50) {
      suggestions.push('Need more light')
    } else if (brightness > 230) {
      suggestions.push('Too bright')
    }

    // Simple blur detection (sampling center area)
    let edgeStrength = 0
    let count = 0
    for (let y = 1; y < 99; y += 5) {
      for (let x = 1; x < 99; x += 5) {
        const idx = (y * 100 + x) * 4
        const center = data[idx]
        const top = data[((y - 1) * 100 + x) * 4]
        const bottom = data[((y + 1) * 100 + x) * 4]
        const left = data[(y * 100 + (x - 1)) * 4]
        const right = data[(y * 100 + (x + 1)) * 4]
        edgeStrength += Math.abs(4 * center - top - bottom - left - right)
        count++
      }
    }
    const blur = edgeStrength / count

    // Blur is informational only - webcam quality varies too much
    if (blur < 3) {
      suggestions.push('Try to hold steady')
    }

    // Ready to capture based on distance and brightness
    // Allow 'unknown' distance for facial/body lesions where hand isn't visible
    // Blur varies too much across webcams to be informational only
    const readyToCapture =
      (distance === 'good' || distance === 'unknown') &&
      brightness >= 50 &&
      brightness <= 230 &&
      blur >= 3 // Minimum blur threshold - must not be severely blurred

    return {
      distance,
      handDetected,
      blur,
      brightness,
      readyToCapture,
      suggestions: suggestions.length > 0 ? suggestions : ['Looking good!'],
      landmarks
    }
  }, [])

  useEffect(() => {
    initialize()
  }, [initialize])

  return { analyzeFrame }
}
