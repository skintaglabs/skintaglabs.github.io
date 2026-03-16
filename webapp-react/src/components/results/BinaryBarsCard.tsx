import { useEffect, useState } from 'react'

interface BinaryBarsCardProps {
  benign: number
  malignant: number
}

export function BinaryBarsCard({ benign, malignant }: BinaryBarsCardProps) {
  const [animatedBenign, setAnimatedBenign] = useState(0)
  const [animatedMalignant, setAnimatedMalignant] = useState(0)

  useEffect(() => {
    const timeout = setTimeout(() => {
      setAnimatedBenign(benign * 100)
      setAnimatedMalignant(malignant * 100)
    }, 200)
    return () => clearTimeout(timeout)
  }, [benign, malignant])

  return (
    <div className="bg-[var(--color-surface)] border rounded-[var(--radius-lg)] p-6">
      <h3 className="text-[17px] font-semibold mb-6">Classification Probabilities</h3>

      <div className="space-y-6">
        <div>
          <div className="flex justify-between items-baseline mb-2">
            <span className="text-[17px] font-medium">Benign</span>
            <span className="text-[15px] text-[var(--color-text-muted)]">
              {Math.round(animatedBenign)}%
            </span>
          </div>
          <div className="h-3 bg-[var(--color-surface-alt)] rounded-full overflow-hidden">
            <div
              className="h-full bg-[var(--color-green)] transition-[width] duration-[1000ms]"
              style={{
                width: `${animatedBenign}%`,
                transitionTimingFunction: 'var(--ease-spring)'
              }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between items-baseline mb-2">
            <span className="text-[17px] font-medium">Malignant</span>
            <span className="text-[15px] text-[var(--color-text-muted)]">
              {Math.round(animatedMalignant)}%
            </span>
          </div>
          <div className="h-3 bg-[var(--color-surface-alt)] rounded-full overflow-hidden">
            <div
              className="h-full bg-[var(--color-red)] transition-[width] duration-[1000ms]"
              style={{
                width: `${animatedMalignant}%`,
                transitionTimingFunction: 'var(--ease-spring)'
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
