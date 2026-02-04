import { type ReactNode, useEffect } from 'react'
import { X } from 'lucide-react'

interface ResultsContainerProps {
  showResults: boolean
  children: ReactNode
  onClose: () => void
}

export function ResultsContainer({ showResults, children, onClose }: ResultsContainerProps) {
  useEffect(() => {
    if (!showResults) return

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [showResults, onClose])

  if (!showResults) return null

  return (
    <>
      <div className="fixed inset-0 z-[100] bg-[var(--color-text)]/40" onClick={onClose} />
      <div className="fixed inset-4 z-[100] flex items-center justify-center">
        <div className="bg-[var(--color-surface)] rounded-[var(--radius-lg)] shadow-[var(--shadow-lg)] w-full max-w-3xl max-h-full overflow-y-auto p-6">
          <button
            onClick={onClose}
            className="sticky top-0 float-right w-10 h-10 rounded-full bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-alt)] flex items-center justify-center transition-colors shadow-[var(--shadow)] -mt-2 -mr-2 mb-4"
          >
            <X className="w-5 h-5" />
          </button>
          {children}
        </div>
      </div>
    </>
  )
}
