import { useState, useRef } from 'react'
import { Toaster } from 'sonner'
import { toast } from 'sonner'
import { AppProvider, useAppContext } from '@/contexts/AppContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { useImageValidation } from '@/hooks/useImageValidation'
import { useAnalysis } from '@/hooks/useAnalysis'
import { useAnalysisHistory } from '@/hooks/useAnalysisHistory'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { BottomNav } from '@/components/layout/BottomNav'
import { OnboardingModal } from '@/components/layout/OnboardingModal'
import { ThemeToggle } from '@/components/layout/ThemeToggle'
import { UploadZone } from '@/components/upload/UploadZone'
import { PreviewCard } from '@/components/upload/PreviewCard'
import { ImageCropper } from '@/components/upload/ImageCropper'
import { ResultsContainer } from '@/components/results/ResultsContainer'
import { SkeletonResults } from '@/components/results/SkeletonResults'
import { Results } from '@/components/results/Results'
import { HistoryView } from '@/components/history/HistoryView'

function AppContent() {
  const { state, setSelectedFile, clearImage, setShowResults, setShowCropper } = useAppContext()
  const { validateImage } = useImageValidation()
  const { analyze } = useAnalysis()
  const { saveAnalysis } = useAnalysisHistory()
  const [currentView, setCurrentView] = useState<'upload' | 'history'>('upload')
  const cameraInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (file: File, previewUrl: string) => {
    setSelectedFile(file, previewUrl)
  }

  const handleAnalyze = async () => {
    if (!state.selectedFile) return

    const results = await analyze(state.selectedFile)

    if (results) {
      try {
        await saveAnalysis(state.selectedFile, results, state.selectedFile.name)
        toast.success('Analysis saved to history', { duration: 2000 })
      } catch {
        console.error('Failed to save to history')
      }
    }
  }

  const handleCloseResults = () => {
    setShowResults(false)
  }

  const handleAnalyzeAnother = () => {
    clearImage()
    setShowResults(false)
  }

  const handleShowCropper = () => {
    setShowCropper(true)
  }

  const handleCropComplete = async (croppedBlob: Blob, croppedUrl: string) => {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl)
    }

    const croppedFile = new File([croppedBlob], state.selectedFile?.name || 'cropped-image.jpg', {
      type: 'image/jpeg'
    })

    setSelectedFile(croppedFile, croppedUrl)
    setShowCropper(false)
  }

  const handleCancelCrop = () => {
    setShowCropper(false)
  }

  const handleCameraClick = () => {
    cameraInputRef.current?.click()
  }

  const handleCameraInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const isValid = await validateImage(file)
      if (isValid) {
        const previewUrl = URL.createObjectURL(file)
        setSelectedFile(file, previewUrl)
        if (currentView === 'history') {
          setCurrentView('upload')
        }
      }
    }
    e.target.value = ''
  }

  return (
    <div className="min-h-screen flex flex-col pb-16">
      <OnboardingModal />

      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleCameraInputChange}
        className="hidden"
      />

      <div className="fixed top-4 right-4 z-50">
        <ThemeToggle />
      </div>

      <div className="flex-1 w-full max-w-3xl mx-auto px-4">
        <Header />

        <main className="pb-8">
          {currentView === 'history' ? (
            <HistoryView />
          ) : (
            <>
              {!state.selectedFile && (
                <UploadZone onFileSelect={handleFileSelect} />
              )}

              {state.selectedFile && state.previewUrl && !state.isAnalyzing && !state.showResults && !state.showCropper && (
                <PreviewCard
                  file={state.selectedFile}
                  previewUrl={state.previewUrl}
                  onClear={clearImage}
                  onAnalyze={handleAnalyze}
                  onCrop={handleShowCropper}
                />
              )}

              {state.selectedFile && state.previewUrl && state.showCropper && (
                <ImageCropper
                  imageUrl={state.previewUrl}
                  onCropComplete={handleCropComplete}
                  onCancel={handleCancelCrop}
                />
              )}

              {state.isAnalyzing && <SkeletonResults />}

              {state.results && (
                <ResultsContainer showResults={state.showResults} onClose={handleCloseResults}>
                  <Results results={state.results} onAnalyzeAnother={handleAnalyzeAnother} />
                </ResultsContainer>
              )}
            </>
          )}
        </main>
      </div>

      <BottomNav
        currentView={currentView}
        onNavigate={setCurrentView}
        onCameraClick={handleCameraClick}
      />

      <Footer />

      <Toaster
        position="top-center"
        toastOptions={{
          style: {
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius)',
            boxShadow: 'var(--shadow-lg)',
            fontFamily: 'inherit'
          }
        }}
      />
    </div>
  )
}

function App() {
  return (
    <ThemeProvider>
      <AppProvider>
        <AppContent />
      </AppProvider>
    </ThemeProvider>
  )
}

export default App
