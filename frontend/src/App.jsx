import React, { useState } from 'react'
import ProbBars from './components/ProbBars.jsx'
import Descriptors from './components/Descriptors.jsx'
import logo from './assets/logo2.png'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export default function App() {
  const [file, setFile] = useState(null)
  const [mode, setMode] = useState('live') // 'live' | 'samples'
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [preview, setPreview] = useState(null)

  const onFileChange = (e) => {
    const f = e.target.files?.[0]
    setError('')
    setResult(null)
    if (f) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    setError('')
    setResult(null)
    if (f) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
    }
  }

  const runSamplePredict = async (filename) => {
    try {
      setLoading(true);
      setError('');
      setResult(null);
      const folder = filename.startsWith('normal_') ? 'normal' : 'suspicious';
      setPreview(`${API_URL}/samples-static/${folder}/${filename}`);

      const resp = await fetch(`${API_URL}/predict?source=sample&name=${filename}`, {
        method: 'POST'
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`Server ${resp.status}: ${txt}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  const predict = async () => {
    try {
      if (!file) { setError('Please choose an image first.'); return }
      setLoading(true); setError(''); setResult(null)

      const fd = new FormData()
      fd.append('file', file)

      const resp = await fetch(`${API_URL}/predict?source=live`, {
        method: 'POST',
        body: fd
      })

      if (!resp.ok) {
        const txt = await resp.text()
        throw new Error(`Server ${resp.status}: ${txt}`)
      }
      const data = await resp.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError('')
  }

  return (
    <div className="container">
      <header className="app-header">
        <img src={logo} alt="SoniSight logo" className="app-logo" />
        <h1>SoniSight (Prototype)</h1>
        <p className="subtitle">
          Upload a breast ultrasound image → OpenCV identifies areas of concern and AI estimates risk levels.
        </p>
      </header>

      <section className="controls">
        <div className="mode">
          <label>
            <input
              type="radio"
              name="mode"
              value="live"
              checked={mode === 'live'}
              onChange={() => setMode('live')}
            />
            Live
          </label>
          <label>
            <input
              type="radio"
              name="mode"
              value="samples"
              checked={mode === 'samples'}
              onChange={() => { setMode('samples'); setFile(null); setPreview(null); setResult(null); }}
            />
            Sample Images
          </label>
        </div>

        {mode === 'live' && (
          <div className="uploader"
               onDragOver={(e)=>e.preventDefault()}
               onDrop={onDrop}>
            <input id="file" type="file" accept="image/*" onChange={onFileChange} />
            <label htmlFor="file" className="dropzone">
              {preview
                ? 'Change image'
                : 'Click to choose or drag & drop an ultrasound image'}
            </label>
          </div>
        )}

        {mode === 'samples' && (
          <div className="sample-grid">
            {['normal_01.png', 'normal_02.png', 'normal_03.png', 'normal_04.png', 'suspicious_01.png', 'suspicious_02.png', 'suspicious_03.png', 'suspicious_04.png'].map(name => {
              const folder = name.startsWith('normal_') ? 'normal' : 'suspicious';
              return (
                <div
                  key={name}
                  className="sample-item"
                  onClick={() => {
                    setError('');
                    setResult(null);
                    setFile(null);
                    runSamplePredict(name);
                  }}
                >
                  <img
                    src={`${API_URL}/samples-static/${folder}/${name}`}
                    alt={name}
                    className="sample"
                  />
                  <div className="sample-label">
                    {(() => {
                      const base = name.replace(/\.[^/.]+$/, '');     // strip extension
                      const match = base.match(/(\d+)/);              // get trailing number
                      const n = match ? match[1].replace(/^0+/, '') : '';
                      const category = folder === 'normal' ? 'Normal' : 'Suspicious';
                      return n ? `${category} (${n})` : category;
                    })()}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="buttons">
          <button onClick={predict} disabled={(mode === 'live' && !file) || loading}>
            {loading ? 'Analyzing…' : 'Analyze Image'}
          </button>
          <button className="ghost" onClick={reset} disabled={loading && !result}>
            Reset
          </button>
        </div>

        {error && <div className="error">{error}</div>}
      </section>

      <section className="results">
        {(preview || (result && result.overlay_png_base64)) && (
          <div className="images">
            {preview && (
              <figure>
                <img src={preview} alt="uploaded" />
                <figcaption>Original</figcaption>
              </figure>
            )}
            {result?.overlay_png_base64 && (
              <figure>
                <img
                  src={`data:image/png;base64,${result.overlay_png_base64}`}
                  alt="overlay"
                />
                <figcaption>Overlay</figcaption>
              </figure>
            )}
          </div>
        )}

        {result && (
          <>
            <div className="row">
              <div className="card">
                <h3>Probabilities</h3>
                <ProbBars probs={result.probabilities} />
              </div>
              <div className="card">
                <h3>Rationale</h3>
                <p>{result.rationale || '—'}</p>
              </div>
            </div>

            <div className="card">
              <h3>Descriptors</h3>
              <Descriptors descriptors={result.descriptors} />
            </div>
          </>
        )}
      </section>

      <footer>
        <span className="disclaimer">
          Prototype for educational purposes and not for clinical use.
        </span>
      </footer>
    {loading && (
      <div
        role="alert"
        aria-live="polite"
        aria-busy="true"
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.35)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}
      >
        <div
          style={{
            background: 'rgba(18, 24, 38, 0.92)',
            padding: '20px 24px',
            borderRadius: 12,
            boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 12,
            minWidth: 220
          }}
        >
          <svg
            width="56"
            height="56"
            viewBox="0 0 50 50"
            style={{ display: 'block' }}
          >
            <circle
              cx="25"
              cy="25"
              r="20"
              fill="none"
              stroke="#00e6c8"
              strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray="90 150"
            >
              <animate
                attributeName="stroke-dashoffset"
                values="0;-124"
                dur="1.2s"
                repeatCount="indefinite"
              />
              <animateTransform
                attributeName="transform"
                type="rotate"
                from="0 25 25"
                to="360 25 25"
                dur="1s"
                repeatCount="indefinite"
              />
            </circle>
          </svg>
          <div style={{ color: '#d8f7f2', fontWeight: 600 }}>
            Analyzing image…
          </div>
        </div>
      </div>
    )}
    </div>
  )
}