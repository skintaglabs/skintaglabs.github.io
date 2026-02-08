import { type ButtonHTMLAttributes, forwardRef } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius-full)] font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        dark: 'bg-[var(--color-text)] text-[var(--color-surface)] shadow-[var(--shadow)] hover:shadow-[var(--shadow-lg)] active:scale-[0.98]',
        ghost: 'hover:bg-[var(--color-surface-alt)] active:scale-[0.98]',
        outline: 'border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[var(--shadow-sm)] hover:shadow-[var(--shadow)] active:scale-[0.98]'
      },
      size: {
        default: 'h-[48px] px-6',
        sm: 'h-[40px] px-4 text-[15px]',
        lg: 'h-[56px] px-8 text-[19px]',
        icon: 'h-[48px] w-[48px]'
      }
    },
    defaultVariants: {
      variant: 'dark',
      size: 'default'
    }
  }
)

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

export { Button }
