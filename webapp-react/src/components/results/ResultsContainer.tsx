import { type ReactNode } from 'react'
import { X } from 'lucide-react'

interface ResultsContainerProps {
  showResults: boolean
  children: ReactNode
  onClose: () => void
}

export function ResultsContainer({ showResults, children, onClose }: ResultsContainerProps) {
  if (!showResults) return null

  return (
    <div className="fixed inset-0 z-[100] bg-[var(--color-bg)] flex flex-col overflow-y-auto">
      <div className="flex-1 w-full max-w-3xl mx-auto px-4 py-6 pb-24">
        <button
          onClick={onClose}
          className="fixed top-4 right-4 w-10 h-10 rounded-full bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-alt)] flex items-center justify-center transition-colors shadow-[var(--shadow)]"
        >
          <X className="w-5 h-5" />
        </button>
        {children}
      </div>
    </div>
  )
}
