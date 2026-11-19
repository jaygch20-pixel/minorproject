import { useRef, useState } from 'react'
import { ImagePlus, LoaderCircle, Trash2, UploadCloud } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { uploadImage } from '../services/api'

const MAX_FILE_SIZE = 10 * 1024 * 1024

export default function UploadPage() {
  const inputRef = useRef(null)
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const validateFile = (candidate) => {
    if (!candidate) return 'Please choose an image file.'
    if (!['image/jpeg', 'image/jpg', 'image/png'].includes(candidate.type)) {
      return 'Only JPG and PNG image files are accepted.'
    }
    if (candidate.size > MAX_FILE_SIZE) {
      return 'Please upload an image under 10MB.'
    }
    return ''
  }

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0]
    if (!selected) return

    const validationMessage = validateFile(selected)
    if (validationMessage) {
      setError(validationMessage)
      setFile(null)
      setPreview('')
      return
    }

    setError('')
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError('Choose a valid image before analyzing.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const result = await uploadImage(file)
      localStorage.setItem('latestAnalysisId', result.id)
      navigate('/results')
    } catch (err) {
      setError(err.message || 'Image analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-stack narrow">
      <section className="panel">
        <div className="section-header">
          <h2>Skin image upload</h2>
        </div>

        <div
          className="dropzone"
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            const droppedFile = event.dataTransfer.files?.[0]
            if (!droppedFile) return

            const validationMessage = validateFile(droppedFile)
            if (validationMessage) {
              setError(validationMessage)
              return
            }

            setFile(droppedFile)
            setPreview(URL.createObjectURL(droppedFile))
            setError('')
          }}
        >
          <UploadCloud size={38} />
          <h3>Drag and drop a skin image</h3>
          <p>Accepts JPG, JPEG, and PNG up to 10MB.</p>
          <button type="button" className="secondary-button">Browse files</button>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={handleFileChange} hidden />
        </div>

        {error && <div className="message error">{error}</div>}

        {preview && (
          <div className="image-preview-card">
            <img src={preview} alt="Selected lesion preview" />
            <div className="image-meta">
              <div>
                <label>Selected file</label>
                <p>{file?.name}</p>
              </div>
              <div>
                <label>Dimensions</label>
                <p>{file ? `${Math.round(file.size / 1024)} KB` : '—'}</p>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={() => {
                  setFile(null)
                  setPreview('')
                  setError('')
                  inputRef.current.value = ''
                }}
              >
                <Trash2 size={15} /> Remove
              </button>
            </div>
          </div>
        )}

        <div className="footer-actions">
          <button type="button" className="primary-button wide" onClick={handleAnalyze} disabled={!file || loading}>
            {loading ? (
              <>
                <LoaderCircle className="spinner" size={16} />
                Analyzing...
              </>
            ) : (
              <>
                <ImagePlus size={16} /> Analyze Image
              </>
            )}
          </button>
        </div>
      </section>
    </div>
  )
}
