import { useEffect, useRef } from 'react'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

interface HandOverlayProps {
  videoElement: HTMLVideoElement
  landmarks: NormalizedLandmark[][]
  state: 'too_far' | 'good' | 'too_close' | 'unknown'
}

export function HandOverlay({ videoElement, landmarks, state }: HandOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || !videoElement || landmarks.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!

    canvas.width = videoElement.videoWidth
    canvas.height = videoElement.videoHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const getColors = () => {
      switch (state) {
        case 'too_far':
          return {
            fill: 'rgba(251, 146, 60, 0.15)', // Orange - softer than red
            stroke: '#fb923c',
            glow: 'rgba(251, 146, 60, 0.3)'
          }
        case 'too_close':
          return {
            fill: 'rgba(59, 130, 246, 0.15)', // Blue
            stroke: '#3b82f6',
            glow: 'rgba(59, 130, 246, 0.3)'
          }
        case 'good':
          return {
            fill: 'rgba(34, 197, 94, 0.15)', // Green
            stroke: '#22c55e',
            glow: 'rgba(34, 197, 94, 0.3)'
          }
        default:
          return {
            fill: 'rgba(156, 163, 175, 0.15)', // Gray
            stroke: '#9ca3af',
            glow: 'rgba(156, 163, 175, 0.3)'
          }
      }
    }

    const colors = getColors()

    landmarks.forEach((handLandmarks) => {
      const xs = handLandmarks.map(l => l.x * canvas.width)
      const ys = handLandmarks.map(l => l.y * canvas.height)
      const minX = Math.min(...xs)
      const maxX = Math.max(...xs)
      const minY = Math.min(...ys)
      const maxY = Math.max(...ys)

      const padding = 20
      const x = minX - padding
      const y = minY - padding
      const width = (maxX - minX) + padding * 2
      const height = (maxY - minY) + padding * 2

      ctx.shadowBlur = 20
      ctx.shadowColor = colors.glow
      ctx.fillStyle = colors.fill
      ctx.fillRect(x, y, width, height)
      ctx.strokeStyle = colors.stroke
      ctx.lineWidth = 3
      ctx.strokeRect(x, y, width, height)
      ctx.shadowBlur = 0

      ctx.strokeStyle = colors.stroke
      ctx.lineWidth = 2
      ctx.globalAlpha = 0.4

      const outline = [
        handLandmarks[0],
        handLandmarks[5],
        handLandmarks[9],
        handLandmarks[13],
        handLandmarks[17],
        handLandmarks[20],
        handLandmarks[16],
        handLandmarks[12],
        handLandmarks[8],
        handLandmarks[4],
        handLandmarks[0]
      ]

      ctx.beginPath()
      ctx.moveTo(outline[0].x * canvas.width, outline[0].y * canvas.height)
      outline.forEach((landmark) => {
        ctx.lineTo(landmark.x * canvas.width, landmark.y * canvas.height)
      })
      ctx.stroke()

      ctx.globalAlpha = 1
    })
  }, [videoElement, landmarks, state])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ objectFit: 'cover', transform: 'scaleX(-1)' }}
    />
  )
}
