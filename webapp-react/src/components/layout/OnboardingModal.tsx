import { useState, useEffect } from 'react'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { Camera, CheckCircle, Sun, Image as ImageIcon } from 'lucide-react'

const ONBOARDING_KEY = 'skintag-onboarding-seen'

const tips = [
  {
    icon: Camera,
    title: 'Take a Clear Photo',
    description: 'Use your phone camera or upload an existing image of the skin lesion you want to analyze.'
  },
  {
    icon: Sun,
    title: 'Good Lighting is Key',
    description: 'Take photos in bright, natural light. Avoid shadows and glare for the most accurate analysis.'
  },
  {
    icon: ImageIcon,
    title: 'Dermoscopic Images Work Best',
    description: 'If available, dermoscopic images provide the highest accuracy. Regular close-up photos also work.'
  },
  {
    icon: CheckCircle,
    title: 'Review Your Results',
    description: 'Get instant risk assessment, ABCDE warning signs, and recommendations. Always consult a healthcare provider.'
  }
]

export function OnboardingModal() {
  const [isOpen, setIsOpen] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)

  useEffect(() => {
    const hasSeenOnboarding = localStorage.getItem(ONBOARDING_KEY)
    if (!hasSeenOnboarding) {
      const timer = setTimeout(() => setIsOpen(true), 500)
      return () => clearTimeout(timer)
    }
  }, [])

  const handleClose = () => {
    localStorage.setItem(ONBOARDING_KEY, 'true')
    setIsOpen(false)
  }

  const handleNext = () => {
    if (currentStep < tips.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      handleClose()
    }
  }

  const handleSkip = () => {
    handleClose()
  }

  const currentTip = tips[currentStep]
  const Icon = currentTip.icon

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetContent className="sm:max-w-md sm:mx-auto sm:my-auto sm:h-auto sm:rounded-[var(--radius-lg)]">
        <div className="flex flex-col h-full sm:h-auto">
          <div className="flex-1 flex flex-col items-center justify-center text-center px-4 py-8 sm:py-12">
            <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-[var(--color-surface-alt)] flex items-center justify-center mb-6">
              <Icon className="w-8 h-8 sm:w-10 sm:h-10 text-[var(--color-accent-warm)]" />
            </div>

            <h2 className="text-[22px] sm:text-[24px] leading-tight font-semibold mb-3" style={{ fontFamily: "'Instrument Serif', serif" }}>
              {currentTip.title}
            </h2>

            <p className="text-[15px] sm:text-[16px] text-[var(--color-text-secondary)] leading-relaxed max-w-sm">
              {currentTip.description}
            </p>

            <div className="flex gap-2 mt-6">
              {tips.map((_, index) => (
                <div
                  key={index}
                  className={`h-1.5 rounded-full transition-all ${
                    index === currentStep
                      ? 'w-8 bg-[var(--color-accent-warm)]'
                      : 'w-1.5 bg-[var(--color-border)]'
                  }`}
                />
              ))}
            </div>
          </div>

          <div className="flex gap-3 p-4 sm:p-6">
            {currentStep < tips.length - 1 ? (
              <>
                <Button onClick={handleSkip} variant="ghost" className="flex-1">
                  Skip
                </Button>
                <Button onClick={handleNext} className="flex-1">
                  Next
                </Button>
              </>
            ) : (
              <Button onClick={handleClose} className="w-full">
                Get Started
              </Button>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
