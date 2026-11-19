import { NavLink, Route, Routes } from 'react-router-dom'
import { Activity, BrainCircuit, Camera, FileText, History, Info, ShieldAlert } from 'lucide-react'
import HomePage from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'
import HistoryPage from './pages/HistoryPage'
import AboutPage from './pages/AboutPage'
import ARPage from './pages/ARPage'

const navItems = [
  { to: '/', label: 'Home', icon: Activity },
  { to: '/upload', label: 'Upload', icon: Camera },
  { to: '/history', label: 'History', icon: History },
  { to: '/about', label: 'About', icon: Info },
  { to: '/ar', label: 'AR Preview', icon: BrainCircuit },
]

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-mark">
            <ShieldAlert size={20} />
          </div>
          <div>
            <div className="brand-title">DermaVision AI</div>
            <div className="brand-subtitle">Experimental research prototype · not a diagnostic tool</div>
          </div>
        </div>

        <nav className="main-nav" aria-label="Main navigation">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Icon size={16} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="page-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/ar" element={<ARPage />} />
        </Routes>
      </main>
    </div>
  )
}
