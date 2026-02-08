export interface TriageCategoryInfo {
  label: string
  probability: number
}

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
  triage_categories?: {
    malignant: TriageCategoryInfo
    inflammatory: TriageCategoryInfo
    benign: TriageCategoryInfo
  }
}

export interface AppState {
  selectedFile: File | null
  previewUrl: string | null
  isAnalyzing: boolean
  results: AnalysisResult | null
  showResults: boolean
  showCropper: boolean
  showWebcam: boolean
}
