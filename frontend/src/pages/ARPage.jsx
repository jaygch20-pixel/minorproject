import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, Eye, EyeOff, Info, ScanLine, SwitchCamera } from 'lucide-react'
import { uploadRealtimeFrame } from '../services/api'

const SAMPLE_INTERVAL_MS = 1500

function probabilityText(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'Unavailable'
}

export default function ARPage() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const inFlightRef = useRef(false)
  const intervalRef = useRef(null)
  const resultRef = useRef(null)
  const [devices, setDevices] = useState([])
  const [deviceId, setDeviceId] = useState('')
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraStatus, setCameraStatus] = useState('Stopped')
  const [analysisStatus, setAnalysisStatus] = useState('Waiting')
  const [error, setError] = useState('')
  const [backendStatus, setBackendStatus] = useState('Unknown')
  const [lastUpdate, setLastUpdate] = useState(null)
  const [result, setResult] = useState(null)
  const [showBoundary, setShowBoundary] = useState(true)
  const [showPanel, setShowPanel] = useState(true)
  const [showLabels, setShowLabels] = useState(true)

  const stopCamera = useCallback(() => {
    if (intervalRef.current) window.clearInterval(intervalRef.current)
    intervalRef.current = null
    inFlightRef.current = false
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraActive(false)
    setCameraStatus('Stopped')
    setAnalysisStatus('Stopped')
    resultRef.current = null
    setResult(null)
    setLastUpdate(null)
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }, [])

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return
    const all = await navigator.mediaDevices.enumerateDevices()
    setDevices(all.filter((item) => item.kind === 'videoinput'))
  }, [])

  const drawOverlay = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return

    const rect = video.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(rect.width * dpr)
    canvas.height = Math.round(rect.height * dpr)
    canvas.style.width = `${rect.width}px`
    canvas.style.height = `${rect.height}px`
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, rect.width, rect.height)

    const active = resultRef.current
    if (!active || !showBoundary) return
    if (!active.segmentation?.is_usable || !active.segmentation?.mask_image_data) return

    const maskImage = new window.Image()
    maskImage.onload = () => {
      const maskWidth = active.segmentation.mask_size?.[0] || 224
      const maskHeight = active.segmentation.mask_size?.[1] || 224
      const sourceWidth = active.preprocessing?.original_size?.[0] || video.videoWidth
      const sourceHeight = active.preprocessing?.original_size?.[1] || video.videoHeight
      const videoScale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight)
      const renderedWidth = video.videoWidth * videoScale
      const renderedHeight = video.videoHeight * videoScale
      const offsetX = (rect.width - renderedWidth) / 2
      const offsetY = (rect.height - renderedHeight) / 2
      const layout = active.preprocessing?.resize_layout

      const toStage = (maskX, maskY) => {
        let sourceX
        let sourceY
        if (layout) {
          sourceX = ((maskX - layout.left) / layout.resized_width) * sourceWidth
          sourceY = ((maskY - layout.top) / layout.resized_height) * sourceHeight
        } else {
          sourceX = (maskX / maskWidth) * sourceWidth
          sourceY = (maskY / maskHeight) * sourceHeight
        }
        return {
          x: offsetX + (sourceX / sourceWidth) * renderedWidth,
          y: offsetY + (sourceY / sourceHeight) * renderedHeight,
        }
      }

      ctx.save()
      ctx.strokeStyle = '#48f7d6'
      ctx.fillStyle = 'rgba(72, 247, 214, 0.14)'
      ctx.lineWidth = 3
      ctx.shadowColor = '#48f7d6'
      ctx.shadowBlur = 8
      const contours = active.segmentation.contour_points || []
      let minX = maskWidth
      let minY = maskHeight
      let maxX = -1
      let maxY = -1
      contours.forEach((contour) => {
        if (!Array.isArray(contour) || contour.length < 3) return
        ctx.beginPath()
        contour.forEach(([x, y], index) => {
          minX = Math.min(minX, x)
          minY = Math.min(minY, y)
          maxX = Math.max(maxX, x)
          maxY = Math.max(maxY, y)
          const point = toStage(x, y)
          if (index === 0) ctx.moveTo(point.x, point.y)
          else ctx.lineTo(point.x, point.y)
        })
        ctx.closePath()
        ctx.stroke()
      })
      ctx.restore()

      if (maxX >= 0) {
        const topLeft = toStage(minX, minY)
        const bottomRight = toStage(maxX, maxY)
        ctx.save()
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1.5
        ctx.setLineDash([7, 5])
        ctx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y)
        ctx.restore()

        if (showLabels) {
          ctx.save()
          ctx.fillStyle = 'rgba(6, 24, 38, 0.92)'
          ctx.strokeStyle = '#48f7d6'
          ctx.lineWidth = 1
          ctx.beginPath()
          ctx.roundRect(topLeft.x + 8, Math.max(8, topLeft.y - 34), 178, 26, 13)
          ctx.fill()
          ctx.stroke()
          ctx.fillStyle = '#ffffff'
          ctx.font = '700 12px Inter, Segoe UI, sans-serif'
          ctx.fillText('LESION BOUNDARY DETECTED', topLeft.x + 17, Math.max(26, topLeft.y - 16))
          ctx.restore()
        }
      }
    }
    maskImage.onerror = () => {}
    maskImage.src = active.segmentation.mask_image_data
  }, [showBoundary, showLabels])

  const sampleFrame = useCallback(async () => {
    const video = videoRef.current
    if (!cameraActive || inFlightRef.current || !video?.videoWidth || !video?.videoHeight) return
    inFlightRef.current = true
    setAnalysisStatus('Analyzing frame')
    const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight))
    const width = Math.max(64, Math.round(video.videoWidth * scale))
    const height = Math.max(64, Math.round(video.videoHeight * scale))
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    canvas.getContext('2d').drawImage(video, 0, 0, width, height)

    canvas.toBlob(async (blob) => {
      if (!blob) {
        setError('A live frame could not be captured.')
        setAnalysisStatus('Frame capture failed')
        inFlightRef.current = false
        return
      }
      const file = new File([blob], `live-frame-${Date.now()}.jpg`, { type: 'image/jpeg' })
      try {
        const response = await uploadRealtimeFrame(file)
        resultRef.current = response
        setResult(response)
        setLastUpdate(new Date())
        setBackendStatus('Connected')
        setAnalysisStatus(response.status === 'success' ? 'Live result updated' : 'Result unavailable')
        setError('')
        drawOverlay()
      } catch (requestError) {
        setBackendStatus(requestError.name === 'AbortError' ? 'Timed out' : 'Offline/error')
        setAnalysisStatus('Analysis failed')
        setError(requestError.message || 'Live frame analysis failed. The camera remains active.')
      } finally {
        inFlightRef.current = false
      }
    }, 'image/jpeg', 0.82)
  }, [cameraActive, drawOverlay])

  const startCamera = useCallback(async (preferredDeviceId = '') => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not supported by this browser.')
      return
    }
    if (!window.isSecureContext) {
      setError('Camera access requires HTTPS or localhost.')
      return
    }
    stopCamera()
    setCameraStatus('Requesting permission')
    try {
      const stream = await navigator.mediaDevices.getUserMedia(preferredDeviceId
        ? { audio: false, video: { deviceId: { exact: preferredDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } } }
        : { audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      const settings = stream.getVideoTracks()[0]?.getSettings?.() || {}
      if (settings.deviceId) setDeviceId(settings.deviceId)
      setCameraActive(true)
      setCameraStatus('Live')
      setBackendStatus('Checking')
      await refreshDevices()
    } catch (requestError) {
      const messages = {
        NotAllowedError: 'Camera permission was denied.',
        NotFoundError: 'No camera device was found.',
        NotReadableError: 'The camera is unavailable or in use.',
        OverconstrainedError: 'The requested camera is unavailable.',
        SecurityError: 'Camera access was blocked by browser security.',
      }
      setError(messages[requestError?.name] || 'The camera could not be started.')
      setCameraStatus('Unavailable')
    }
  }, [refreshDevices, stopCamera])

  useEffect(() => {
    refreshDevices()
    startCamera()
    return stopCamera
  }, [refreshDevices, startCamera, stopCamera])

  useEffect(() => {
    if (!cameraActive) return
    if (intervalRef.current) window.clearInterval(intervalRef.current)
    intervalRef.current = window.setInterval(sampleFrame, SAMPLE_INTERVAL_MS)
    const initialTimeout = window.setTimeout(sampleFrame, 500)
    const redraw = () => drawOverlay()
    window.addEventListener('resize', redraw)
    window.addEventListener('orientationchange', redraw)
    return () => {
      window.clearInterval(intervalRef.current)
      window.clearTimeout(initialTimeout)
      window.removeEventListener('resize', redraw)
      window.removeEventListener('orientationchange', redraw)
    }
  }, [cameraActive, sampleFrame, drawOverlay])

  useEffect(() => {
    if (cameraActive) drawOverlay()
  }, [cameraActive, result, showBoundary, showLabels, showPanel, drawOverlay])

  const switchCamera = async () => {
    if (devices.length < 2) {
      setError('No additional camera was found.')
      return
    }
    const index = devices.findIndex((item) => item.deviceId === deviceId)
    await startCamera(devices[(index >= 0 ? index + 1 : 0) % devices.length].deviceId)
  }

  const features = result?.features?.traditional_features
  return (
    <div className="page-stack ar-live-page">
      <section className="panel ar-live-shell">
        <div className="scanner-header">
          <div>
            <span className="scanner-eyebrow">DERMAVISION REAL-TIME CV</span>
            <h2>Live lesion scanner</h2>
          </div>
          <div className="scanner-statuses">
            <span className={cameraActive ? 'live-badge active' : 'live-badge'}><span className="dot" /> LIVE</span>
            <span className="live-badge scanning"><ScanLine size={13} /> {analysisStatus}</span>
          </div>
        </div>

        <div className="scanner-warning">
          <Info size={16} />
          Research prototype only. Smartphone images are not clinically validated. The model missed all melanoma cases in the held-out report at 0.50.
        </div>
        {error && <div className="message error scanner-error" role="alert">{error}</div>}

        <div className="scanner-stage">
          <video ref={videoRef} className="scanner-video" playsInline muted />
          <canvas ref={canvasRef} className="scanner-overlay" />
          {cameraActive && (
            <div className="scanner-frame" aria-label="Position the lesion inside the scanning frame">
              <span>Position lesion here</span>
            </div>
          )}
          {!cameraActive && (
            <div className="scanner-placeholder"><Camera size={38} /><span>{cameraStatus}</span></div>
          )}

          {showPanel && (
            <aside className="scanner-hud">
              <div className="hud-section hud-primary">
                <span className="hud-title">CNN ANALYSIS</span>
                <strong>{result?.prediction || 'Awaiting result'}</strong>
                <div><span>Melanoma</span><b>{probabilityText(result?.probabilities?.melanoma)}</b></div>
                <div><span>Non-melanoma</span><b>{probabilityText(result?.probabilities?.non_melanoma)}</b></div>
              </div>
              <div className="hud-section">
                <span className="hud-title">COMPUTER VISION</span>
                <div><span>Segmentation</span><b>{result?.segmentation?.status || 'Awaiting result'}</b></div>
                <div><span>Area fraction</span><b>{typeof features?.lesion_area_fraction === 'number' ? `${(features.lesion_area_fraction * 100).toFixed(2)}%` : 'Unavailable'}</b></div>
                <div><span>Circularity</span><b>{typeof features?.circularity === 'number' ? features.circularity.toFixed(3) : 'Unavailable'}</b></div>
                <div><span>Mean RGB</span><b>{features?.mean_rgb ? features.mean_rgb.map((v) => v.toFixed(0)).join(', ') : 'Unavailable'}</b></div>
              </div>
              <div className="hud-section hud-system">
                <span className="hud-title">SYSTEM</span>
                <div><span>Camera</span><b>{cameraStatus}</b></div>
                <div><span>Backend</span><b>{backendStatus}</b></div>
                <div><span>Updated</span><b>{lastUpdate ? lastUpdate.toLocaleTimeString() : 'Never'}</b></div>
              </div>
            </aside>
          )}
        </div>

        <div className="scanner-controls">
          {!cameraActive ? (
            <button type="button" className="primary-button" onClick={() => startCamera(deviceId)}><Camera size={16} /> Start Scanning</button>
          ) : (
            <button type="button" className="danger-button" onClick={stopCamera}><CameraOff size={16} /> Stop Scanning</button>
          )}
          {devices.length > 1 && (
            <button type="button" className="secondary-button" onClick={switchCamera}><SwitchCamera size={16} /> Switch Camera</button>
          )}
          <button type="button" className="secondary-button" onClick={() => setShowBoundary((value) => !value)}>
            {showBoundary ? <EyeOff size={15} /> : <Eye size={15} />} Boundary
          </button>
          <button type="button" className="secondary-button" onClick={() => setShowLabels((value) => !value)}>
            {showLabels ? <EyeOff size={15} /> : <Eye size={15} />} Labels
          </button>
          <button type="button" className="secondary-button" onClick={() => setShowPanel((value) => !value)}>
            <Eye size={15} /> {showPanel ? 'Hide panel' : 'Show panel'}
          </button>
        </div>

        <div className="scanner-guidance">
          <span>Keep steady</span><span>Use bright even light</span><span>Avoid blur</span><span>Do not digitally zoom</span>
        </div>
      </section>
    </div>
  )
}
