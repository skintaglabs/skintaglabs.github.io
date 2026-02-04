import type { AnalysisResult } from '@/types'
import { TierCard } from './TierCard'
import { RiskDisplay } from './RiskDisplay'
import { ABCDEGrid } from './ABCDEGrid'
import { ConditionCard } from './ConditionCard'
import { BinaryBarsCard } from './BinaryBarsCard'
import { CTAActions } from './CTAActions'
import { DisclaimerBanner } from '@/components/layout/DisclaimerBanner'

interface ResultsProps {
  results: AnalysisResult
}

export function Results({ results }: ResultsProps) {
  return (
    <div className="space-y-6">
      <div id="results-capture" className="space-y-6">
        <TierCard
          tier={results.urgency_tier}
          confidence={results.confidence}
          recommendation={results.recommendation}
        />

        <RiskDisplay
          score={results.risk_score}
          tier={results.urgency_tier}
        />

        <ABCDEGrid />

        {results.condition_estimate && results.condition_probabilities && (
          <ConditionCard
            topCondition={results.condition_estimate}
            conditions={results.condition_probabilities}
          />
        )}

        <BinaryBarsCard
          benign={results.probabilities.benign}
          malignant={results.probabilities.malignant}
        />

        <DisclaimerBanner />
      </div>

      <CTAActions tier={results.urgency_tier} results={results} />
    </div>
  )
}
