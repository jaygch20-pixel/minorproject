const API_BASE = 'http://localhost:8000'

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}))
    throw new Error(errorBody.detail || 'Request failed')
  }

  return response.json()
}

export async function getHealth() {
  return apiFetch('/api/health')
}

export async function getDatasetSummary() {
  return apiFetch('/api/dataset/summary')
}

export async function getEvaluationMetrics() {
  return apiFetch('/api/evaluation/metrics')
}

export async function uploadImage(file, source = 'upload') {
  const formData = new FormData()
  formData.append('file', file, file.name || 'camera-capture.jpg')
  formData.append('source', source)

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  })

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(data.detail || 'Upload failed')
  }

  return data
}

export async function uploadRealtimeFrame(file) {
  const formData = new FormData()
  formData.append('file', file, file.name || 'live-frame.jpg')
  formData.append('source', 'camera')
  formData.append('realtime', 'true')

  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), 12000)
  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(data.detail || 'Live frame analysis failed')
    return data
  } finally {
    window.clearTimeout(timeoutId)
  }
}

export async function getHistory() {
  return apiFetch('/api/history')
}

export async function getHistoryRecord(id) {
  return apiFetch(`/api/history/${id}`)
}

export async function deleteHistoryRecord(id) {
  return apiFetch(`/api/history/${id}`, { method: 'DELETE' })
}
