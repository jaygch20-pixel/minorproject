import { useEffect, useState } from 'react'
import { Delete, FileText, Trash2 } from 'lucide-react'
import { deleteHistoryRecord, getHistory } from '../services/api'

export default function HistoryPage() {
  const [items, setItems] = useState([])

  async function loadHistory() {
    try {
      const result = await getHistory()
      setItems(result.items || [])
    } catch (error) {
      console.error('Could not load history:', error)
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  const handleDelete = async (id) => {
    try {
      await deleteHistoryRecord(id)
      setItems((current) => current.filter((item) => item.id !== id))
    } catch (error) {
      console.error('Delete failed:', error)
    }
  }

  return (
    <div className="page-stack narrow">
      <section className="panel">
        <div className="section-header">
          <h2>Analysis history</h2>
        </div>

        {items.length === 0 ? (
          <div className="empty-state">
            <FileText size={18} />
            No previous analyses are stored yet.
          </div>
        ) : (
          <div className="history-list">
            {items.map((item) => (
              <article className="history-item" key={item.id}>
                <img src={item.image_data || '/placeholder.png'} alt={item.filename} />
                <div className="history-copy">
                  <strong>{item.filename || 'Uploaded image'}</strong>
                  <div>{item.status === 'model_unavailable' ? 'Prediction unavailable' : item.prediction || 'No prediction'}</div>
                  <small>{item.source === 'camera' ? 'Camera capture' : 'Manual upload'}</small>
                  <small>{new Date(item.created_at).toLocaleString()}</small>
                </div>
                <div className="history-actions">
                  <button type="button" className="ghost-button small" onClick={() => handleDelete(item.id)}>
                    <Trash2 size={14} /> Delete
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
