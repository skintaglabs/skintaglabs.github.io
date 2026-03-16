import { useEffect, useState } from 'react'

interface Condition {
  condition: string
  probability: number
}

interface ConditionCardProps {
  topCondition: string
  conditions: Condition[]
}

const conditionDescriptions: Record<string, string> = {
  'melanoma': 'The most serious type of skin cancer that can spread to other organs',
  'basal cell carcinoma': 'The most common form of skin cancer, rarely spreads but should be treated',
  'squamous cell carcinoma': 'A common skin cancer from sun-damaged cells, treatable when caught early',
  'actinic keratosis': 'A rough, scaly patch caused by sun damage that may develop into skin cancer',
  'melanocytic nevus': 'A common mole, typically benign',
  'seborrheic keratosis': 'A harmless waxy skin growth, often age-related',
  'dermatofibroma': 'A common benign fibrous nodule, usually harmless',
  'vascular lesion': 'Blood vessel abnormality in the skin, usually benign',
  'non-neoplastic': 'Inflammatory or reactive skin condition, not a tumor',
  'other/unknown': 'Unclassified lesion -- consult a dermatologist for evaluation',
  // Legacy aliases
  'benign keratosis': 'A harmless skin growth, often age-related',
  'nevus': 'A common mole, typically benign',
}

export function ConditionCard({ topCondition, conditions }: ConditionCardProps) {
  const [animatedBars, setAnimatedBars] = useState<number[]>([])

  useEffect(() => {
    const timeout = setTimeout(() => {
      setAnimatedBars(conditions.map(c => c.probability * 100))
    }, 200)
    return () => clearTimeout(timeout)
  }, [conditions])

  return (
    <div className="bg-[var(--color-surface)] border rounded-[var(--radius-lg)] p-6">
      <h3 className="text-[17px] font-semibold mb-1">Most Likely Condition</h3>
      <p
        className="text-[28px] leading-tight font-semibold mb-2"
        style={{ fontFamily: "'Instrument Serif', serif" }}
      >
        {topCondition}
      </p>
      <p className="text-[15px] text-[var(--color-text-secondary)] mb-6 leading-relaxed">
        {conditionDescriptions[topCondition.toLowerCase()] || 'Consult a dermatologist for diagnosis.'}
      </p>

      <div className="space-y-3">
        {conditions.slice(0, 3).map((condition, index) => (
          <div key={condition.condition}>
            <div className="flex justify-between items-baseline mb-1.5">
              <span className="text-[15px] font-medium">{condition.condition}</span>
              <span className="text-[13px] text-[var(--color-text-muted)]">
                {Math.round(condition.probability * 100)}%
              </span>
            </div>
            <div className="h-2 bg-[var(--color-surface-alt)] rounded-full overflow-hidden">
              <div
                className="h-full bg-[var(--color-accent-warm)] transition-[width] duration-[1000ms]"
                style={{
                  width: `${animatedBars[index] || 0}%`,
                  transitionTimingFunction: 'var(--ease-spring)'
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
