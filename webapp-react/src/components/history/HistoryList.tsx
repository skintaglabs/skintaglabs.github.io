import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { History, Trash2, ChevronRight, AlertCircle } from 'lucide-react'
import { useAnalysisHistory, type HistoryEntry } from '@/hooks/useAnalysisHistory'
import { toast } from 'sonner'

interface HistoryListProps {
  onViewEntry: (entry: HistoryEntry) => void
}

const tierColors = {
  low: 'var(--color-green)',
  moderate: 'var(--color-amber)',
  high: 'var(--color-red)'
}

export function HistoryList({ onViewEntry }: HistoryListProps) {
  const { history, isLoading, deleteAnalysis, clearHistory } = useAnalysisHistory()
  const [showConfirmClear, setShowConfirmClear] = useState(false)

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await deleteAnalysis(id)
      toast.success('Analysis deleted')
    } catch {
      toast.error('Failed to delete analysis')
    }
  }

  const handleClearAll = async () => {
    try {
      await clearHistory()
      toast.success('History cleared')
      setShowConfirmClear(false)
    } catch {
      toast.error('Failed to clear history')
    }
  }

  if (isLoading) {
    return (
      <div className="text-center py-12 text-[var(--color-text-muted)]">
        Loading history...
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div className="text-center py-12">
        <History className="w-12 h-12 text-[var(--color-text-muted)] mx-auto mb-4" />
        <p className="text-[17px] text-[var(--color-text-secondary)]">
          No saved analyses yet
        </p>
        <p className="text-[15px] text-[var(--color-text-muted)] mt-1">
          Your analysis history will appear here
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-[24px] font-semibold" style={{ fontFamily: "'Instrument Serif', serif" }}>
          Analysis History
        </h2>
        <Button
          onClick={() => setShowConfirmClear(true)}
          variant="ghost"
          size="sm"
          className="text-[var(--color-red)]"
        >
          Clear All
        </Button>
      </div>

      {showConfirmClear && (
        <div className="p-4 border-[var(--color-red)] bg-[var(--color-red-bg)] rounded-[var(--radius-lg)] border">
          <div className="flex items-start gap-3 mb-4">
            <AlertCircle className="w-5 h-5 text-[var(--color-red)] flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-[15px] font-medium">Clear all history?</p>
              <p className="text-[13px] text-[var(--color-text-muted)] mt-1">
                This will permanently delete all saved analyses.
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <Button onClick={() => setShowConfirmClear(false)} variant="outline" className="flex-1">
              Cancel
            </Button>
            <Button onClick={handleClearAll} className="flex-1 bg-[var(--color-red)] hover:opacity-90">
              Delete All
            </Button>
          </div>
        </div>
      )}

      <div className="space-y-3">
        {history.map((entry) => (
          <div
            key={entry.id}
            className="overflow-hidden cursor-pointer hover:shadow-[var(--shadow-lg)] transition-all rounded-[var(--radius-lg)] bg-[var(--color-surface)] border"
            onClick={() => onViewEntry(entry)}
          >
            <div className="flex gap-4 p-4">
              <div className="w-20 h-20 rounded-[var(--radius)] overflow-hidden bg-[var(--color-surface-alt)] flex-shrink-0">
                <img
                  src={entry.imageUrl}
                  alt="Analysis"
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2 mb-1">
                  <p className="text-[15px] font-medium truncate">{entry.fileName}</p>
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0 mt-2"
                    style={{ backgroundColor: tierColors[entry.results.urgency_tier] }}
                  />
                </div>

                <p className="text-[13px] text-[var(--color-text-muted)] mb-2">
                  {new Date(entry.timestamp).toLocaleString()}
                </p>

                <div className="flex items-center gap-2">
                  <span
                    className="text-[13px] font-medium capitalize"
                    style={{ color: tierColors[entry.results.urgency_tier] }}
                  >
                    {entry.results.urgency_tier} Risk
                  </span>
                  <span className="text-[13px] text-[var(--color-text-muted)]">
                    • {Math.round(entry.results.risk_score * 100)}%
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2 flex-shrink-0">
                <button
                  onClick={(e) => handleDelete(entry.id, e)}
                  className="w-9 h-9 rounded-full hover:bg-[var(--color-surface-alt)] flex items-center justify-center transition-colors"
                >
                  <Trash2 className="w-4 h-4 text-[var(--color-text-muted)]" />
                </button>
                <ChevronRight className="w-5 h-5 text-[var(--color-text-muted)]" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
