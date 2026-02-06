import { Skeleton } from '@/components/ui/skeleton'

export function SkeletonResults() {
  return (
    <>
      <div className="fixed inset-0 z-[100] bg-black/40" />
      <div className="fixed inset-4 z-[100] flex items-center justify-center">
        <div className="bg-[var(--color-surface)] rounded-[var(--radius-lg)] shadow-[var(--shadow-lg)] w-full max-w-3xl max-h-full overflow-y-auto p-6">
          <div className="space-y-6">
            <div className="space-y-3">
              <Skeleton className="h-8 w-32" />
              <Skeleton className="h-24 w-full" />
            </div>

            <div className="space-y-3">
              <Skeleton className="h-6 w-24" />
              <Skeleton className="h-32 w-full" />
            </div>

            <div className="space-y-3">
              <Skeleton className="h-6 w-40" />
              <div className="grid grid-cols-3 gap-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-20" />
                ))}
              </div>
            </div>

            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        </div>
      </div>
    </>
  )
}
