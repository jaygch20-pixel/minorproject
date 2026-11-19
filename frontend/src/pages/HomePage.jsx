import { useEffect, useState } from 'react'
import { ArrowRight, BrainCircuit, Camera, Clock3, ShieldAlert, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { getHealth, getHistory } from '../services/api'

const featureCards = [
  {
    title: 'AI-Assisted Analysis',
    description: 'Upload a lesion image to run the prototype workflow and review preliminary outputs.',
    icon: BrainCircuit,
  },
  {
    title: 'Visual Explanation',
    description: 'Review model explanations and future saliency overlays as part of the research workflow.',
    icon: Sparkles,
  },
  {
    title: 'AR Visualization',
    description: 'Explore the planned augmented overlay concept with an interactive prototype placeholder.',
    icon: Camera,
  },
]

export default function HomePage() {
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])

  useEffect(() => {
    async function loadHome() {
      try {
        const [healthResponse, historyResponse] = await Promise.all([getHealth(), getHistory()])
        setHealth(healthResponse)
        setHistory(historyResponse.items || [])
      } catch (error) {
        console.error('Home page failed to load:', error)
      }
    }

    loadHome()
  }, [])

  return (
    <div className="page-stack">
      <section className="hero-card">
        <div>
          <span className="eyebrow">Research prototype</span>
          <h1>Augmented analysis for early skin disease assessment.</h1>
          <p>
            DermaVision AI helps researchers explore melanoma classification workflows using uploaded skin images.
            This prototype is designed for academic investigation and education, not medical diagnosis.
          </p>

          <div className="hero-actions">
            <Link className="primary-button" to="/upload">
              Analyze Skin Image
              <ArrowRight size={18} />
            </Link>
            <Link className="secondary-button" to="/about">
              Learn More
            </Link>
          </div>

          <div className="status-row">
            <div className={`status-pill ${health?.status === 'ok' ? 'online' : ''}`}>
              <span className="dot" />
              {health?.status === 'ok' ? 'Backend online' : 'Backend unavailable'}
            </div>
            <div className="status-note">
              {health?.model_available ? `CNN available: ${health.model_name}` : 'No trained CNN; predictions are unavailable'}
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <div className="visual-card">
            <div className="mini-chart" />
            <div className="mini-panel">
              <span>Dataset summary</span>
              <strong>ISIC 2020</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="feature-grid">
        {featureCards.map(({ title, description, icon: Icon }) => (
          <article className="info-card" key={title}>
            <div className="icon-circle">
              <Icon size={22} />
            </div>
            <h3>{title}</h3>
            <p>{description}</p>
          </article>
        ))}
      </section>

      <section className="history-panel">
        <div className="section-header">
          <h2>Recent analysis history</h2>
          <Link to="/history">View all</Link>
        </div>

        {history.length === 0 ? (
          <div className="empty-state small">
            <Clock3 size={18} />
            No saved analyses yet. Upload an image to begin.
          </div>
        ) : (
          <div className="history-list compact">
            {history.slice(0, 3).map((entry) => (
              <div className="history-item compact-item" key={entry.id}>
                <img src={entry.image_data || '/placeholder.png'} alt={entry.filename} />
                <div>
                  <strong>{entry.filename || 'Uploaded image'}</strong>
                  <div>{entry.status === 'model_unavailable' ? 'Prediction unavailable' : entry.prediction || 'No prediction'}</div>
                  <small>{entry.source === 'camera' ? 'Camera capture' : 'Manual upload'}</small>
                </div>
                <span>{new Date(entry.created_at).toLocaleDateString()}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="disclaimer-box">
        <ShieldAlert size={18} />
        <p>
          This project is an experimental research prototype. It is not a medical diagnostic system and must not be
          used to diagnose or rule out melanoma.
        </p>
      </section>
    </div>
  )
}
