import { createContext, useContext, useState, type ReactNode } from 'react'
import type { AppState, AnalysisResult } from '@/types'

interface AppContextType {
  state: AppState
  setSelectedFile: (file: File | null, previewUrl: string | null) => void
  clearImage: () => void
  setResults: (results: AnalysisResult | null) => void
  setIsAnalyzing: (isAnalyzing: boolean) => void
  setShowResults: (show: boolean) => void
  setShowCropper: (show: boolean) => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>({
    selectedFile: null,
    previewUrl: null,
    isAnalyzing: false,
    results: null,
    showResults: false,
    showCropper: false
  })

  const setSelectedFile = (file: File | null, previewUrl: string | null) => {
    setState(prev => ({ ...prev, selectedFile: file, previewUrl }))
  }

  const clearImage = () => {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl)
    }
    setState({
      selectedFile: null,
      previewUrl: null,
      isAnalyzing: false,
      results: null,
      showResults: false,
      showCropper: false
    })
  }

  const setResults = (results: AnalysisResult | null) => {
    setState(prev => ({ ...prev, results, showResults: results !== null }))
  }

  const setIsAnalyzing = (isAnalyzing: boolean) => {
    setState(prev => ({ ...prev, isAnalyzing }))
  }

  const setShowResults = (show: boolean) => {
    setState(prev => ({ ...prev, showResults: show }))
  }

  const setShowCropper = (show: boolean) => {
    setState(prev => ({ ...prev, showCropper: show }))
  }

  return (
    <AppContext.Provider value={{ state, setSelectedFile, clearImage, setResults, setIsAnalyzing, setShowResults, setShowCropper }}>
      {children}
    </AppContext.Provider>
  )
}

export function useAppContext() {
  const context = useContext(AppContext)
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider')
  }
  return context
}
