import { useCallback } from 'react'

interface QualityResult {
  passed: boolean
  score: number
  issues: string[]
}

export function useImageQuality() {
  const analyzeQuality = useCallback(async (file: File): Promise<QualityResult> => {
    try {
      const image = await createImageBitmap(file)
      const canvas = document.createElement('canvas')
      canvas.width = image.width
      canvas.height = image.height

      const ctx = canvas.getContext('2d', { willReadFrequently: true })!
      ctx.drawImage(image, 0, 0)

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const data = imageData.data

      const issues: string[] = []
      let score = 100

      // 1. Check for sufficient brightness
      let totalBrightness = 0
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i]
        const g = data[i + 1]
        const b = data[i + 2]
        totalBrightness += (r + g + b) / 3
      }
      const avgBrightness = totalBrightness / (data.length / 4)

      if (avgBrightness < 40) {
        issues.push('Image is too dark')
        score -= 30
      } else if (avgBrightness > 240) {
        issues.push('Image is overexposed')
        score -= 30
      } else if (avgBrightness < 80) {
        issues.push('Lighting could be better')
        score -= 15
      }

      // 2. Check for blur using edge detection (simplified Laplacian)
      const blurScore = calculateBlurScore(imageData)
      if (blurScore < 50) {
        issues.push('Image appears blurry')
        score -= 25
      } else if (blurScore < 100) {
        issues.push('Image could be sharper')
        score -= 10
      }

      // 3. Check for skin tones (heuristic: detect pinkish/brownish hues)
      const hasSkinTones = detectSkinTones(data)
      if (!hasSkinTones) {
        issues.push('No skin detected in image')
        score -= 40
      }

      // 4. Check contrast
      const contrast = calculateContrast(data)
      if (contrast < 20) {
        issues.push('Low contrast - may be hard to analyze')
        score -= 15
      }

      return {
        passed: score >= 40,
        score,
        issues
      }
    } catch (error) {
      console.error('Quality analysis failed:', error)
      return {
        passed: true, // Graceful degradation
        score: 0,
        issues: []
      }
    }
  }, [])

  return { analyzeQuality }
}

function calculateBlurScore(imageData: ImageData): number {
  const data = imageData.data
  const width = imageData.width
  const height = imageData.height

  // Sample every 10th pixel for performance
  let edgeStrength = 0
  let count = 0

  for (let y = 1; y < height - 1; y += 10) {
    for (let x = 1; x < width - 1; x += 10) {
      const idx = (y * width + x) * 4
      const center = data[idx]

      // Simplified Laplacian kernel
      const top = data[((y - 1) * width + x) * 4]
      const bottom = data[((y + 1) * width + x) * 4]
      const left = data[(y * width + (x - 1)) * 4]
      const right = data[(y * width + (x + 1)) * 4]

      const laplacian = Math.abs(4 * center - top - bottom - left - right)
      edgeStrength += laplacian
      count++
    }
  }

  return edgeStrength / count
}

function detectSkinTones(data: Uint8ClampedArray): boolean {
  let skinPixels = 0
  const totalPixels = data.length / 4
  const sampleRate = 50 // Check every 50th pixel for performance

  for (let i = 0; i < data.length; i += 4 * sampleRate) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]

    // Skin tone detection heuristic
    // Skin typically has: R > G > B, R > 95, G > 40, B > 20
    // and certain ratios between channels
    const rg = r - g
    const rb = r - b

    if (r > 95 && g > 40 && b > 20 &&
        rg > 15 && rb > 15 &&
        r > g && g > b) {
      skinPixels++
    }
  }

  const skinPercentage = (skinPixels / (totalPixels / sampleRate)) * 100
  return skinPercentage > 10 // At least 10% skin tones
}

function calculateContrast(data: Uint8ClampedArray): number {
  let min = 255
  let max = 0

  for (let i = 0; i < data.length; i += 4) {
    const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3
    min = Math.min(min, brightness)
    max = Math.max(max, brightness)
  }

  return max - min
}
