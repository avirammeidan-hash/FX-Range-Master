interface Props {
  /** 0–100 */
  pct: number
  /** Label inside gauge center */
  label?: string
  /** Sub-label below gauge */
  sublabel?: string
  /** Threshold tick position (0–100), default 55 */
  threshold?: number
  size?: 'sm' | 'md' | 'lg'
}

const SIZE = { sm: 120, md: 176, lg: 220 }

/**
 * Semicircular confidence / accuracy gauge.
 * Generic — works for ML confidence, win rate, accuracy scores, etc.
 *
 * @example
 * <AccuracyGauge pct={73} label="73%" sublabel="Win rate" threshold={60} />
 */
export default function AccuracyGauge({ pct, label, sublabel, threshold = 55, size = 'md' }: Props) {
  const px = SIZE[size]
  const degrees = (Math.min(100, Math.max(0, pct)) / 100) * 180

  const color =
    pct >= 75 ? '#22c55e' :
    pct >= 55 ? '#f59e0b' :
    pct >= 40 ? '#f97316' : '#ef4444'

  const colorClass =
    pct >= 75 ? 'text-gain' :
    pct >= 55 ? 'text-warn' :
    pct >= 40 ? 'text-orange' : 'text-loss'

  // Threshold marker coords on the arc (viewBox 200×110, radius 80, center 100,100)
  const tAngle = (threshold / 100) * Math.PI
  const tx1 = 100 - 90 * Math.cos(tAngle)
  const ty1 = 100 - 90 * Math.sin(tAngle)
  const tx2 = 100 - 70 * Math.cos(tAngle)
  const ty2 = 100 - 70 * Math.sin(tAngle)

  return (
    <div className="relative flex flex-col items-center" style={{ width: px }}>
      <svg viewBox="0 0 200 110" className="w-full">
        {/* BG arc */}
        <path d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none" stroke="#1e293b" strokeWidth="18" strokeLinecap="round" />
        {/* Value arc */}
        <path d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none" stroke={color} strokeWidth="18" strokeLinecap="round"
          strokeDasharray={`${(degrees / 180) * 251.2} 251.2`}
          style={{ transition: 'stroke-dasharray 0.8s ease' }}
        />
        {/* Threshold marker */}
        <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke="#4f8eff" strokeWidth="2.5" strokeLinecap="round" />
        {/* Tick labels at 0, 25, 50, 75, 100 */}
        {[0, 25, 50, 75, 100].map(tick => {
          const a = (tick / 100) * Math.PI
          const x = 100 - 105 * Math.cos(a)
          const y = 100 - 105 * Math.sin(a)
          return <text key={tick} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
            fill="#4b5563" fontSize="8">{tick}%</text>
        })}
      </svg>
      {/* Center label */}
      <div className="absolute inset-0 flex flex-col items-center justify-end pb-1">
        <span className={`font-bold num ${colorClass} ${size === 'lg' ? 'text-4xl' : size === 'sm' ? 'text-xl' : 'text-3xl'}`}>
          {label ?? `${Math.round(pct)}%`}
        </span>
      </div>
      {sublabel && <p className="text-xs font-medium mt-0.5" style={{ color }}>{sublabel}</p>}
    </div>
  )
}
