import React from 'react'

export default function ProbBars({ probs }) {
  const order = ['normal', 'suspicious']
  const p = order.map(k => ({ k, v: Math.round((probs?.[k] || 0) * 1000)/10 }))
  return (
    <div className="probs">
      {p.map(({k,v}) => (
        <div key={k} className="probrow">
          <div className="label">{k}</div>
          <div className="bar">
            <div className="fill" style={{ width: `${Math.max(0, Math.min(100, v))}%` }} />
          </div>
          <div className="pct">{isFinite(v) ? v.toFixed(1) : '0.0'}%</div>
        </div>
      ))}
    </div>
  )
}