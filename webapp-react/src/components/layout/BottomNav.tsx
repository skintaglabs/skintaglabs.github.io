import { Home, Camera, History } from 'lucide-react'

interface BottomNavProps {
  currentView: 'upload' | 'history'
  onNavigate: (view: 'upload' | 'history') => void
  onCameraClick: () => void
}

export function BottomNav({ currentView, onNavigate, onCameraClick }: BottomNavProps) {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-40 bg-[var(--color-surface)] border-t border-[var(--color-border)] safe-area-pb">
      <div className="relative flex items-center justify-around max-w-3xl mx-auto h-20">
        <button
          onClick={() => onNavigate('upload')}
          className={`flex flex-col items-center justify-center gap-1 min-w-[64px] transition-all active:scale-95 ${
            currentView === 'upload'
              ? 'text-[var(--color-accent-warm)]'
              : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
          }`}
        >
          <Home className="w-6 h-6" />
          <span className="text-[12px] font-medium">Home</span>
        </button>

        <button
          onClick={onCameraClick}
          className="absolute left-1/2 -translate-x-1/2 -top-3"
        >
          <div className="w-16 h-16 rounded-full bg-[var(--color-accent-warm)] shadow-[var(--shadow-lg)] flex items-center justify-center hover:scale-105 active:scale-95 transition-transform">
            <Camera className="w-7 h-7 text-[var(--color-surface)]" />
          </div>
        </button>

        <button
          onClick={() => onNavigate('history')}
          className={`flex flex-col items-center justify-center gap-1 min-w-[64px] transition-all active:scale-95 ${
            currentView === 'history'
              ? 'text-[var(--color-accent-warm)]'
              : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
          }`}
        >
          <History className="w-6 h-6" />
          <span className="text-[12px] font-medium">History</span>
        </button>
      </div>
    </nav>
  )
}
