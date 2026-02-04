import { useEffect, useRef, useState } from 'react'
import { Camera, X, AlertCircle, CheckCircle } from 'lucide-react'
import { useRealtimeValidation } from '@/hooks/useRealtimeValidation'
import { HandOverlay } from './HandOverlay'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

interface WebcamCaptureProps {
  onCapture: (file: File) => void
  onClose: () => void
}

export function WebcamCapture({ onCapture, onClose }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number | undefined>(undefined)
  const [error, setError] = useState<string>('')
  const [isReady, setIsReady] = useState(false)
  const [metrics, setMetrics] = useState<{
    distance: 'too_far' | 'good' | 'too_close' | 'unknown'
    handDetected: boolean
    blur: number
    brightness: number
    readyToCapture: boolean
    suggestions: string[]
    landmarks: NormalizedLandmark[][]
  }>({
    distance: 'unknown',
    handDetected: false,
    blur: 0,
    brightness: 0,
    readyToCapture: false,
    suggestions: ['Starting camera...'],
    landmarks: []
  })

  const { analyzeFrame } = useRealtimeValidation()

  useEffect(() => {
    startWebcam()
    return () => {
      stopWebcam()
    }
  }, [])

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsReady(true)
        startAnalysis()
      }
    } catch (err) {
      console.error('Webcam access error:', err)
      setError('Unable to access webcam. Please check permissions.')
    }
  }

  const stopWebcam = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
  }

  const startAnalysis = () => {
    const analyze = () => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        const timestamp = performance.now()
        const frameMetrics = analyzeFrame(videoRef.current, timestamp)
        setMetrics(frameMetrics)
      }
      animationFrameRef.current = requestAnimationFrame(analyze)
    }
    analyze()
  }

  const handleCapture = () => {
    if (!videoRef.current || !isReady) return

    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext('2d')!
    ctx.drawImage(video, 0, 0)

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `webcam-${Date.now()}.jpg`, { type: 'image/jpeg' })
        stopWebcam()
        onCapture(file)
      }
    }, 'image/jpeg', 0.95)
  }

  const handleClose = () => {
    stopWebcam()
    onClose()
  }

  const getStatusColor = () => {
    if (metrics.readyToCapture) return 'text-green-500'
    if (metrics.distance === 'too_far' || metrics.distance === 'too_close') return 'text-yellow-500'
    return 'text-gray-400'
  }

  const getStatusIcon = () => {
    if (metrics.readyToCapture) return <CheckCircle className="w-5 h-5" />
    return <AlertCircle className="w-5 h-5" />
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center">
      <div className="relative w-full h-full max-w-5xl flex flex-col">
        <button
          onClick={handleClose}
          className="absolute top-4 right-4 z-10 w-10 h-10 rounded-full bg-black/50 text-white hover:bg-black/70 flex items-center justify-center transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex-1 flex items-center justify-center relative">
          {error ? (
            <div className="text-center text-white space-y-4">
              <p className="text-lg">{error}</p>
              <button
                onClick={handleClose}
                className="px-6 py-3 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          ) : (
            <>
              <div className="relative rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="max-w-full max-h-[70vh] rounded-lg"
                />

                {isReady && videoRef.current && metrics.landmarks.length > 0 && (
                  <HandOverlay
                    videoElement={videoRef.current}
                    landmarks={metrics.landmarks}
                    state={metrics.distance}
                  />
                )}

                {!isReady && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <div className="text-white text-lg">Starting camera...</div>
                  </div>
                )}

                {isReady && (
                  <>
                    {/* Guide overlay */}
                    <div className="absolute inset-0 pointer-events-none">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div
                          className={`w-64 h-64 border-4 rounded-full transition-colors ${
                            metrics.readyToCapture
                              ? 'border-green-500'
                              : metrics.distance === 'good'
                              ? 'border-yellow-500'
                              : 'border-white/30'
                          }`}
                        />
                      </div>
                    </div>

                    {/* Live metrics overlay */}
                    <div className="absolute top-4 left-4 space-y-2">
                      <div className={`flex items-center gap-2 px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm ${getStatusColor()}`}>
                        {getStatusIcon()}
                        <span className="text-sm font-medium">
                          {metrics.readyToCapture ? 'Ready to capture!' : 'Adjusting...'}
                        </span>
                      </div>

                      {metrics.suggestions.map((suggestion, i) => (
                        <div
                          key={i}
                          className="px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm text-white text-sm"
                        >
                          {suggestion}
                        </div>
                      ))}

                      {/* Debug metrics */}
                      <div className="px-3 py-2 rounded-lg bg-black/40 backdrop-blur-sm text-white/60 text-xs space-y-1">
                        <div>Distance: {metrics.distance}</div>
                        <div>Blur: {metrics.blur.toFixed(0)}</div>
                        <div>Brightness: {metrics.brightness.toFixed(0)}</div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </div>

        {isReady && !error && (
          <div className="p-6 flex justify-center gap-4">
            <button
              onClick={handleCapture}
              className={`w-16 h-16 rounded-full transition-all flex items-center justify-center shadow-lg ${
                metrics.readyToCapture
                  ? 'bg-green-500 hover:bg-green-600 scale-110'
                  : 'bg-white hover:bg-gray-100'
              } active:scale-95`}
            >
              <Camera className={`w-8 h-8 ${metrics.readyToCapture ? 'text-white' : 'text-gray-900'}`} />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
