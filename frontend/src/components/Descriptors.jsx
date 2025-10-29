import React from 'react'

export default function Descriptors({ descriptors }) {
  if (!descriptors) return null
  const entries = Object.entries(descriptors)
  return (
    <table className="desc">
      <tbody>
        {entries.map(([k,v]) => (
          <tr key={k}>
            <td className="k">{k}</td>
            <td className="v">{String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}