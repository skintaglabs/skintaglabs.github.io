const warningSign = [
  { letter: 'A', word: 'Asymmetry', hint: 'One half unlike the other' },
  { letter: 'B', word: 'Border', hint: 'Irregular or poorly defined' },
  { letter: 'C', word: 'Color', hint: 'Varied shades of brown, black' },
  { letter: 'D', word: 'Diameter', hint: 'Larger than 6mm (pencil eraser)' },
  { letter: 'E', word: 'Evolving', hint: 'Changing size, shape, or color' }
]

export function ABCDEGrid() {
  return (
    <div>
      <h3 className="text-[17px] font-semibold mb-4">ABCDE Warning Signs</h3>
      <div className="grid grid-cols-3 sm:grid-cols-5 gap-3">
        {warningSign.map(({ letter, word, hint }) => (
          <div
            key={letter}
            className="bg-[var(--color-surface)] border rounded-[var(--radius)] p-3 sm:p-4 transition-all hover:shadow-[var(--shadow)] hover:-translate-y-0.5 cursor-default"
          >
            <div
              className="text-[28px] sm:text-[32px] leading-none font-semibold mb-1.5 sm:mb-2"
              style={{ fontFamily: "'Instrument Serif', serif" }}
            >
              {letter}
            </div>
            <div className="text-[14px] sm:text-[15px] font-medium mb-1 break-words">{word}</div>
            <div className="text-[12px] sm:text-[13px] text-[var(--color-text-muted)] leading-snug break-words">
              {hint}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
