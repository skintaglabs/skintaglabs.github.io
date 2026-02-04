import { Info } from 'lucide-react'
import { useState } from 'react'
import { Sheet, SheetContent } from '@/components/ui/sheet'

export function Header() {
  const [showInfo, setShowInfo] = useState(false)

  return (
    <>
      <header className="relative w-full max-w-3xl mx-auto pt-[var(--space-4)] pb-[var(--space-3)] px-[var(--space-2)]">
        <button
          onClick={() => setShowInfo(true)}
          className="absolute right-[var(--space-2)] top-[var(--space-4)] w-12 h-12 rounded-full hover:bg-[var(--color-surface-alt)] flex items-center justify-center transition-all active:scale-95"
          aria-label="About"
        >
          <Info className="w-5 h-5 text-[var(--color-text-muted)]" />
        </button>

        <div className="flex flex-col items-center">
          <h1 className="text-[40px] sm:text-[48px] leading-[1.1] tracking-tight mb-[var(--space-2)]" style={{ fontFamily: "'Instrument Serif', serif" }}>
            Skin<span className="text-[var(--color-accent-warm)]" style={{ fontStyle: 'italic' }}>Tag</span>
          </h1>
          <p className="text-[15px] sm:text-[17px] text-[var(--color-text-secondary)] text-center max-w-md">
            AI-powered skin lesion risk assessment
          </p>
        </div>
      </header>

      <Sheet open={showInfo} onOpenChange={setShowInfo}>
        <SheetContent>
          <div className="flex flex-col gap-[var(--space-4)] h-full">
            <div className="pt-[var(--space-3)]">
              <h2 className="text-[28px] leading-tight font-semibold mb-[var(--space-1)]" style={{ fontFamily: "'Instrument Serif', serif" }}>
                About SkinTag
              </h2>
              <p className="text-[15px] text-[var(--color-text-secondary)] leading-relaxed">
                Advanced AI technology for evaluating skin lesion risk.
              </p>
            </div>

            <div className="flex-1 overflow-y-auto -mx-[var(--space-3)] px-[var(--space-3)]">
              <div className="flex flex-col gap-[var(--space-4)]">
                <div className="h-px bg-[var(--color-border)]" />

                <div>
                  <h3 className="text-[17px] font-semibold mb-[var(--space-2)]">Training Data</h3>
                  <p className="text-[15px] text-[var(--color-text-secondary)] leading-relaxed">
                    Trained on 10,015 clinical dermoscopy images from multiple medical institutions.
                  </p>
                </div>

                <div className="bg-[var(--color-red-bg)] border border-[var(--color-red)] rounded-[var(--radius-lg)] p-[var(--space-3)]">
                  <h3 className="text-[15px] font-semibold text-[var(--color-red)] mb-[var(--space-2)]">
                    Medical Disclaimer
                  </h3>
                  <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed">
                    This tool is for educational purposes only and should not replace professional medical advice.
                    Always consult a healthcare provider for skin concerns.
                  </p>
                </div>

                <div>
                  <h3 className="text-[17px] font-semibold mb-[var(--space-3)]">How It Works</h3>
                  <div className="flex flex-col gap-[var(--space-3)]">
                    <div className="flex gap-[var(--space-2)]">
                      <div className="w-6 h-6 rounded-full bg-[var(--color-accent-warm)] text-[var(--color-surface)] flex items-center justify-center flex-shrink-0 text-[13px] font-medium mt-0.5">
                        1
                      </div>
                      <p className="text-[15px] text-[var(--color-text-secondary)] leading-relaxed">
                        Upload or capture a photo of your skin lesion
                      </p>
                    </div>
                    <div className="flex gap-[var(--space-2)]">
                      <div className="w-6 h-6 rounded-full bg-[var(--color-accent-warm)] text-[var(--color-surface)] flex items-center justify-center flex-shrink-0 text-[13px] font-medium mt-0.5">
                        2
                      </div>
                      <p className="text-[15px] text-[var(--color-text-secondary)] leading-relaxed">
                        AI analyzes the image for melanoma risk
                      </p>
                    </div>
                    <div className="flex gap-[var(--space-2)]">
                      <div className="w-6 h-6 rounded-full bg-[var(--color-accent-warm)] text-[var(--color-surface)] flex items-center justify-center flex-shrink-0 text-[13px] font-medium mt-0.5">
                        3
                      </div>
                      <p className="text-[15px] text-[var(--color-text-secondary)] leading-relaxed">
                        Receive risk assessment and recommendations
                      </p>
                    </div>
                    <div className="flex gap-[var(--space-2)]">
                      <div className="w-6 h-6 rounded-full bg-[var(--color-accent-warm)] text-[var(--color-surface)] flex items-center justify-center flex-shrink-0 text-[13px] font-medium mt-0.5">
                        4
                      </div>
                      <p className="text-[15px] text-[var(--color-text-secondary)] leading-relaxed">
                        History saved locally for tracking changes
                      </p>
                    </div>
                  </div>
                </div>

                <div className="h-[var(--space-4)]" />
              </div>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
