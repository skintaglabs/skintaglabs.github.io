import type { AnalysisResult } from '@/types'
import { TierCard } from './TierCard'
import { RiskDisplay } from './RiskDisplay'
import { ABCDEGrid } from './ABCDEGrid'
import { TriageCategoryCard } from './TriageCategoryCard'
import { ConditionCard } from './ConditionCard'
import { BinaryBarsCard } from './BinaryBarsCard'
import { CTAActions } from './CTAActions'
import { DisclaimerBanner } from '@/components/layout/DisclaimerBanner'
import { useAppContext } from '@/contexts/AppContext'

interface ResultsProps {
  results: AnalysisResult
  onAnalyzeAnother?: () => void
}

export function Results({ results, onAnalyzeAnother }: ResultsProps) {
  const { state } = useAppContext()

  return (
    <div className="space-y-6">
      {state.previewUrl && (
        <div className="flex justify-center result-item">
          <img
            src={state.previewUrl}
            alt="Analyzed image"
            className="w-48 h-48 object-cover rounded-lg border-2 border-[var(--color-border)] preview-image"
          />
        </div>
      )}

      <div className="result-item">
        <TierCard
          tier={results.urgency_tier}
          confidence={results.confidence}
          recommendation={results.recommendation}
        />
      </div>

      <div className="result-item">
        <RiskDisplay
          score={results.risk_score}
          tier={results.urgency_tier}
        />
      </div>

      {results.triage_categories && (
        <div className="result-item">
          <TriageCategoryCard categories={results.triage_categories} />
        </div>
      )}

      <div className="result-item">
        <ABCDEGrid />
      </div>

      {results.condition_estimate && results.condition_probabilities && (
        <div className="result-item">
          <ConditionCard
            topCondition={results.condition_estimate}
            conditions={results.condition_probabilities}
          />
        </div>
      )}

      <div className="result-item">
        <BinaryBarsCard
          benign={results.probabilities.benign}
          malignant={results.probabilities.malignant}
        />
      </div>

      <div className="result-item">
        <DisclaimerBanner />
      </div>

      <CTAActions tier={results.urgency_tier} results={results} onAnalyzeAnother={onAnalyzeAnother} />
    </div>
  )
}
