import * as React from 'react'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

const Sheet = DialogPrimitive.Root
const SheetPortal = DialogPrimitive.Portal

const SheetOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    className={cn(
      'fixed inset-0 z-50 bg-[var(--color-text)]/40 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      className
    )}
    {...props}
    ref={ref}
  />
))
SheetOverlay.displayName = DialogPrimitive.Overlay.displayName

const SheetContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <SheetPortal>
    <SheetOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'fixed left-0 right-0 md:left-auto md:right-0 md:w-[420px] bottom-0 z-50 flex flex-col bg-[var(--color-surface)] px-[var(--space-3)] pb-[var(--space-3)] pt-[var(--space-2)] shadow-[var(--shadow-lg)] rounded-t-[var(--radius-lg)] md:rounded-tl-[var(--radius-lg)] max-h-[85vh] data-[state=open]:animate-slideUp',
        className
      )}
      {...props}
    >
      <DialogPrimitive.Close className="absolute right-[var(--space-3)] top-[var(--space-3)] w-10 h-10 rounded-full hover:bg-[var(--color-surface-alt)] flex items-center justify-center transition-all active:scale-95">
        <X className="h-4 w-4 text-[var(--color-text-muted)]" />
        <span className="sr-only">Close</span>
      </DialogPrimitive.Close>
      {children}
    </DialogPrimitive.Content>
  </SheetPortal>
))
SheetContent.displayName = DialogPrimitive.Content.displayName

export { Sheet, SheetContent }
