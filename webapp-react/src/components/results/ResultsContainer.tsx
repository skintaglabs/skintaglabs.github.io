import { type ReactNode, useEffect, useCallback } from 'react'
import { X } from 'lucide-react'

interface ResultsContainerProps {
  showResults: boolean
  children: ReactNode
  onClose: () => void
}

export function ResultsContainer({ showResults, children, onClose }: ResultsContainerProps) {
  const handleEscape = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault()
      onClose()
    }
  }, [onClose])

  useEffect(() => {
    if (!showResults) return

    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [showResults, handleEscape])

  if (!showResults) return null

  return (
    <>
      <div className="fixed inset-0 z-[100] bg-black/40" onClick={onClose} />
      <div className="fixed inset-4 z-[101] flex items-center justify-center pointer-events-none">
        <div className="relative bg-[var(--color-surface)] rounded-[var(--radius-lg)] shadow-[var(--shadow-lg)] w-full max-w-3xl max-h-full overflow-y-auto p-6 pointer-events-auto">
          <button
            onClick={onClose}
            className="absolute top-3 right-3 z-10 w-10 h-10 rounded-full bg-[var(--color-text)]/80 hover:bg-[var(--color-text)] text-[var(--color-surface)] flex items-center justify-center transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
          {children}
        </div>
      </div>
    </>
  )
}
