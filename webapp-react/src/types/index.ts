export interface AnalysisResult {
  risk_score: number
  urgency_tier: 'low' | 'moderate' | 'high'
  confidence: string
  recommendation: string
  disclaimer: string
  probabilities: {
    benign: number
    malignant: number
  }
  condition_estimate?: string
  condition_probabilities?: Array<{
    condition: string
    probability: number
  }>
}

export interface AppState {
  selectedFile: File | null
  previewUrl: string | null
  isAnalyzing: boolean
  results: AnalysisResult | null
  showResults: boolean
  showCropper: boolean
}

export interface TierConfig {
  low: string
  moderate: string
  high: string
}
