import { type HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('analyzing-skeleton rounded-[var(--radius)]', className)}
      {...props}
    />
  )
}

export { Skeleton }
