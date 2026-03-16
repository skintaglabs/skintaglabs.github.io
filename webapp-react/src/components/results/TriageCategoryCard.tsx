import { useEffect, useState } from 'react'
import type { TriageCategoryInfo } from '@/types'

interface TriageCategoryCardProps {
  categories: {
    malignant: TriageCategoryInfo
    inflammatory: TriageCategoryInfo
    benign: TriageCategoryInfo
  }
}

const CATEGORY_CONFIG = {
  malignant: {
    color: 'var(--color-red)',
    bg: 'var(--color-red-bg, rgba(239, 68, 68, 0.1))',
    description: 'Cancerous or pre-cancerous lesions requiring medical attention',
  },
  inflammatory: {
    color: 'var(--color-amber)',
    bg: 'var(--color-amber-bg, rgba(245, 158, 11, 0.1))',
    description: 'Reactive or inflammatory skin conditions',
  },
  benign: {
    color: 'var(--color-green)',
    bg: 'var(--color-green-bg, rgba(34, 197, 94, 0.1))',
    description: 'Harmless growths unlikely to require treatment',
  },
} as const

type CategoryKey = keyof typeof CATEGORY_CONFIG

const ORDER: CategoryKey[] = ['malignant', 'inflammatory', 'benign']

export function TriageCategoryCard({ categories }: TriageCategoryCardProps) {
  const [animated, setAnimated] = useState<Record<CategoryKey, number>>({
    malignant: 0,
    inflammatory: 0,
    benign: 0,
  })

  useEffect(() => {
    const timeout = setTimeout(() => {
      setAnimated({
        malignant: categories.malignant.probability * 100,
        inflammatory: categories.inflammatory.probability * 100,
        benign: categories.benign.probability * 100,
      })
    }, 200)
    return () => clearTimeout(timeout)
  }, [categories])

  // Find the dominant category
  const dominant = ORDER.reduce((a, b) =>
    categories[a].probability >= categories[b].probability ? a : b
  )

  return (
    <div className="bg-[var(--color-surface)] border rounded-[var(--radius-lg)] p-6">
      <h3 className="text-[17px] font-semibold mb-2">Triage Classification</h3>
      <p className="text-[13px] text-[var(--color-text-muted)] mb-6">
        Probability breakdown across three clinical categories
      </p>

      <div className="space-y-5">
        {ORDER.map((key) => {
          const config = CATEGORY_CONFIG[key]
          const pct = animated[key]
          const isDominant = key === dominant

          return (
            <div key={key}>
              <div className="flex justify-between items-baseline mb-1.5">
                <div className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full inline-block"
                    style={{ backgroundColor: config.color }}
                  />
                  <span className={`text-[15px] ${isDominant ? 'font-semibold' : 'font-medium'}`}>
                    {categories[key].label}
                  </span>
                </div>
                <span className={`text-[15px] tabular-nums ${isDominant ? 'font-semibold' : 'text-[var(--color-text-muted)]'}`}>
                  {Math.round(pct)}%
                </span>
              </div>
              <div className="h-3 bg-[var(--color-surface-alt)] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-[width] duration-[1000ms]"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: config.color,
                    transitionTimingFunction: 'var(--ease-spring)',
                  }}
                />
              </div>
              <p className="text-[12px] text-[var(--color-text-muted)] mt-1">
                {config.description}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
