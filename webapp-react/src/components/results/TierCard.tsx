interface TierCardProps {
  tier: 'low' | 'moderate' | 'high'
  confidence: string
  recommendation: string
}

const tierConfig = {
  low: {
    color: 'var(--color-green)',
    bg: 'var(--color-green-bg)',
    label: 'Low Risk'
  },
  moderate: {
    color: 'var(--color-amber)',
    bg: 'var(--color-amber-bg)',
    label: 'Moderate Risk'
  },
  high: {
    color: 'var(--color-red)',
    bg: 'var(--color-red-bg)',
    label: 'High Risk'
  }
}

export function TierCard({ tier, confidence, recommendation }: TierCardProps) {
  const config = tierConfig[tier]

  return (
    <div
      className="rounded-[var(--radius-lg)] p-4 sm:p-6 border"
      style={{ backgroundColor: config.bg, borderColor: config.color }}
    >
      <div className="flex items-start gap-2 sm:gap-3 mb-3 sm:mb-4">
        <div className="flex items-center gap-2 sm:gap-3 flex-1 min-w-0">
          <div
            className="w-3 h-3 rounded-full flex-shrink-0"
            style={{ backgroundColor: config.color }}
          />
          <h3 className="text-[20px] sm:text-[24px] font-semibold" style={{ color: config.color }}>
            {config.label}
          </h3>
        </div>
        <div className="px-2.5 sm:px-3 py-0.5 sm:py-1 rounded-full bg-[var(--color-surface)] border text-[12px] sm:text-[13px] font-medium whitespace-nowrap flex-shrink-0">
          {confidence} confidence
        </div>
      </div>
      <p className="text-[14px] sm:text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
        {recommendation}
      </p>
    </div>
  )
}
