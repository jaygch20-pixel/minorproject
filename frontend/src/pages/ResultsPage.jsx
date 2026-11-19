import { useEffect, useState } from 'react'
import { AlertTriangle, ArrowLeft, BarChart3, RefreshCw, ShieldAlert } from 'lucide-react'
import { Link } from 'react-router-dom'
import { getEvaluationMetrics, getHistoryRecord } from '../services/api'

const FEATURE_LABELS = {
  lesion_area_px: 'Lesion area (pixels)',
  lesion_area_fraction: 'Lesion area fraction',
  lesion_perimeter_px: 'Lesion perimeter (pixels)',
  circularity: 'Approximate circularity',
  bounding_box_aspect_ratio: 'Bounding-box aspect ratio',
  extent: 'Bounding-box extent',
  eccentricity: 'Approximate eccentricity',
  mean_rgb: 'Mean RGB colour',
  rgb_standard_deviation: 'RGB colour variation',
  texture_contrast: 'Texture contrast',
  texture_homogeneity: 'Texture homogeneity',
}

function formatFeatureValue(value) {
  if (Array.isArray(value)) return value.map((channel) => channel.toFixed(1)).join(', ')
  if (typeof value !== 'number') return 'Unavailable'
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4)
}

function ImageResult({ title, src, alt, unavailableMessage = 'Image unavailable.' }) {
  const [imageError, setImageError] = useState(false)

  useEffect(() => {
    setImageError(false)
  }, [src])

  return (
    <article className="panel analysis-image-card">
      <h3>{title}</h3>
      {src && !imageError
        ? <img src={src} alt={alt} onError={() => setImageError(true)} />
        : <p className="unavailable-note">
          {imageError ? 'Image data could not be decoded.' : unavailableMessage}
        </p>}
    </article>
  )
}

function EvaluationSummary({ title, metrics, unavailableMessage }) {
  if (!metrics) {
    return (
      <section className="panel">
        <div className="section-header"><h2>{title}</h2></div>
        <p className="unavailable-note">{unavailableMessage}</p>
      </section>
    )
  }

  const metricRows = [
    ['Accuracy', metrics.accuracy],
    ['Precision', metrics.precision],
    ['Melanoma sensitivity / recall', metrics.melanoma_sensitivity_recall],
    ['Specificity', metrics.specificity],
    ['F1 score', metrics.f1_score],
    ['ROC-AUC', metrics.roc_auc],
    ['Melanoma-positive examples', metrics.positive_examples],
    ['Non-melanoma examples', metrics.negative_examples],
  ]
  const missedAllMelanoma = metrics.positive_examples > 0
    && metrics.melanoma_sensitivity_recall === 0
    && metrics.confusion_matrix?.[1]?.[1] === 0

  return (
    <section className="panel">
      <div className="section-header"><h2>{title}</h2></div>
      <p className="subtle-note">
        Evaluation source: {metrics.evaluation_source}. Fixed decision threshold: {metrics.threshold ?? 0.5}.
      </p>
      <dl className="feature-list">
        {metricRows.map(([label, value]) => (
          <div className="feature-row" key={label}>
            <dt>{label}</dt>
            <dd>
              {typeof value !== 'number'
                ? 'Unavailable'
                : label === 'Melanoma-positive examples' || label === 'Non-melanoma examples'
                  ? value.toLocaleString()
                  : label === 'ROC-AUC'
                    ? value.toFixed(4)
                    : `${(value * 100).toFixed(2)}%`}
            </dd>
          </div>
        ))}
      </dl>
      {Array.isArray(metrics.confusion_matrix) && metrics.confusion_matrix.length === 2 && (
        <div className="confusion-matrix-wrap">
          <h3>Confusion matrix</h3>
          <p className="subtle-note">
            Rows are actual classes; columns are predicted classes. Positive class: melanoma.
          </p>
          <table className="confusion-matrix">
            <caption>Classification counts</caption>
            <thead>
              <tr>
                <th scope="col">Actual \ Predicted</th>
                <th scope="col">Non-melanoma</th>
                <th scope="col">Melanoma</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Non-melanoma</th>
                <td>{metrics.confusion_matrix[0]?.[0]?.toLocaleString?.() ?? 'Unavailable'}</td>
                <td>{metrics.confusion_matrix[0]?.[1]?.toLocaleString?.() ?? 'Unavailable'}</td>
              </tr>
              <tr>
                <th scope="row">Melanoma</th>
                <td>{metrics.confusion_matrix[1]?.[0]?.toLocaleString?.() ?? 'Unavailable'}</td>
                <td>{metrics.confusion_matrix[1]?.[1]?.toLocaleString?.() ?? 'Unavailable'}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
      {missedAllMelanoma && (
        <div className="warning-box evaluation-warning" role="alert">
          <AlertTriangle size={16} />
          This model missed all {metrics.positive_examples} melanoma-positive cases in this fold at the stated threshold.
        </div>
      )}
    </section>
  )
}

export default function ResultsPage() {
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [currentEvaluationMetrics, setCurrentEvaluationMetrics] = useState(null)
  const [evaluationError, setEvaluationError] = useState('')

  useEffect(() => {
    let active = true
    getEvaluationMetrics()
      .then((response) => {
        if (!active) return
        if (response.status === 'ok') {
          setCurrentEvaluationMetrics(response)
        } else {
          setEvaluationError(response.message || 'Latest evaluation metrics are unavailable.')
        }
      })
      .catch((requestError) => {
        if (active) setEvaluationError(requestError.message || 'Latest evaluation metrics could not be loaded.')
      })

    async function loadResult() {
      const id = localStorage.getItem('latestAnalysisId')
      if (id) {
        try {
          const record = await getHistoryRecord(id)
          if (active) setAnalysis({ ...record.result, ...record, image_data: record.image_data })
        } catch (requestError) {
          if (active) setError(requestError.message || 'Could not load the analysis result.')
        } finally {
          if (active) setLoading(false)
        }
        return
      }

      const legacyItem = localStorage.getItem('latestAnalysis')
      if (legacyItem && active) {
        try {
          setAnalysis(JSON.parse(legacyItem))
        } catch {
          setError('The saved analysis result could not be read. Upload the image again to create a new result.')
        }
      }
      if (active) setLoading(false)
    }
    loadResult()
    return () => { active = false }
  }, [])

  if (loading) {
    return (
      <div className="page-stack narrow">
        <section className="panel empty-state-box" role="status">
          <span className="spinner"><RefreshCw size={20} /></span>
          <p>Loading analysis results...</p>
        </section>
      </div>
    )
  }

  if (!analysis) {
    return (
      <div className="page-stack narrow">
        <section className="panel empty-state-box">
          <AlertTriangle size={20} />
          <h2>{error ? 'Could not load analysis result.' : 'No analysis result available.'}</h2>
          <p>{error || 'Upload an image and run the analysis workflow to see output here.'}</p>
          <Link className="primary-button" to="/upload">
            <ArrowLeft size={16} /> Upload image
          </Link>
        </section>
      </div>
    )
  }

  const probabilities = analysis.probabilities || {}
  const traditionalFeatures = analysis.features?.traditional_features
  const metrics = currentEvaluationMetrics || analysis.evaluation_metrics
  const validationMetrics = metrics?.validation_evaluation
  const segmentation = analysis.segmentation
  const hasForeground = Number.isFinite(segmentation?.foreground_pixels)
    ? segmentation.foreground_pixels > 0
    : typeof segmentation?.area_fraction === 'number' && segmentation.area_fraction > 0
  const segmentationAvailable = segmentation?.is_usable === true && hasForeground
  const segmentationUnavailableMessage = segmentation?.message
    ? `Approximate segmentation unavailable: ${segmentation.message}`
    : 'Approximate segmentation did not produce a usable mask.'

  return (
    <div className="page-stack">
      <section className="result-header">
        <div>
          <span className="eyebrow">Analysis result</span>
          <h1>Experimental evaluation summary</h1>
        </div>
        <div className="tag status-tag">{analysis.model_available ? 'CNN model' : 'Prediction unavailable'}</div>
      </section>

      <p className="subtle-note">
        Segmentation outputs are approximate computer-vision results, not medically verified lesion boundaries.
      </p>
      <section className="analysis-image-grid">
        <ImageResult title="Original image" src={analysis.image_data} alt={analysis.filename || 'Uploaded image'} />
        <ImageResult title="Preprocessed CNN input" src={analysis.preprocessing?.image_data} alt="Preprocessed RGB image" />
        <ImageResult
          title="Approximate binary lesion mask"
          src={segmentationAvailable ? segmentation.mask_image_data : null}
          alt="Approximate binary computer-vision segmentation mask"
          unavailableMessage={segmentationUnavailableMessage}
        />
        <ImageResult
          title="Approximate lesion-boundary overlay"
          src={segmentationAvailable ? segmentation.boundary_overlay_image_data : null}
          alt="Approximate computer-vision lesion boundary overlay"
          unavailableMessage={segmentationUnavailableMessage}
        />
        <ImageResult
          title="Approximate segmented image"
          src={segmentationAvailable ? segmentation.segmented_image_data : null}
          alt="Image segmented using an approximate computer-vision mask"
          unavailableMessage={segmentationUnavailableMessage}
        />
      </section>

      <section className="panel">
        <div className="section-header"><h2>Experimental model output (not a diagnosis)</h2></div>
        {analysis.model_available ? (
          <>
            <div className="prediction-pill">{analysis.prediction}</div>
            <div className="metric-row">
              <label>Top-class score (not clinically calibrated)</label>
              <strong>{analysis.confidence === null ? 'Unavailable' : `${(analysis.confidence * 100).toFixed(1)}%`}</strong>
            </div>
            <p className="subtle-note">Softmax output scores; they are not validated clinical probabilities.</p>
            <p className="subtle-note">Model output scores</p>
            <div className="probability-list">
              {['melanoma', 'non_melanoma'].map((label) => {
                const probability = probabilities[label]
                const hasProbability = typeof probability === 'number' && Number.isFinite(probability)
                return (
                  <div className="probability-item" key={label}>
                    <label>{label === 'melanoma' ? 'Melanoma' : 'Non-melanoma'}</label>
                    {hasProbability
                      ? <div className="bar-track"><span style={{ width: `${probability * 100}%` }} /></div>
                      : <span className="unavailable-note">Unavailable</span>}
                    <strong>{hasProbability ? `${(probability * 100).toFixed(1)}%` : 'Unavailable'}</strong>
                  </div>
                )
              })}
            </div>
          </>
        ) : (
          <p className="unavailable-note">No trained ISIC2020 CNN is available. The app did not generate a prediction.</p>
        )}
        <p className="subtle-note">
          Preprocessed RGB input is classified directly; the approximate segmentation mask is not supplied to the CNN.
        </p>
        <div className="warning-box evaluation-warning" role="alert">
          <AlertTriangle size={16} />
          This is an experimental research prediction, not a diagnosis. A “non_melanoma” result does not rule out
          melanoma or mean a lesion is safe.
        </div>
        {metrics?.melanoma_sensitivity_recall === 0 && (
          <div className="warning-box evaluation-warning" role="alert">
            <AlertTriangle size={16} />
            On the patient-disjoint held-out test fold, this model detected 0 of {metrics.positive_examples} melanoma
            cases at the fixed 0.50 threshold. Do not rely on a negative output.
          </div>
        )}
        <div className="warning-box"><AlertTriangle size={16} />{analysis.warning}</div>
      </section>

      <section className="panel">
        <div className="section-header"><h2>Preprocessing and segmentation</h2></div>
        <div className="analysis-status-list">
          <p><strong>Preprocessing:</strong> {analysis.preprocessing?.status || 'Unavailable'}; {analysis.preprocessing?.input_size?.join(' × ') || 'size unavailable'} pixels; {analysis.preprocessing?.normalization || 'normalization unavailable'}.</p>
          <p><strong>Segmentation:</strong> {segmentation?.status || 'Unavailable'} ({segmentation?.method || 'no method'}). {segmentation?.message || ''}</p>
          <p><strong>Traditional features:</strong> {analysis.features?.reason || analysis.features?.status || 'Unavailable'}</p>
        </div>
      </section>

      <section className="panel">
        <div className="section-header"><h2>Extracted traditional features</h2></div>
        {traditionalFeatures ? (
          <dl className="feature-list">
            {Object.entries(traditionalFeatures).map(([key, value]) => (
              <div className="feature-row" key={key}>
                <dt>{FEATURE_LABELS[key] || key}</dt>
                <dd>{value === null ? 'Unavailable' : formatFeatureValue(value)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="unavailable-note">Lesion measurements are unavailable because no usable lesion mask was produced.</p>
        )}
        <p className="subtle-note">These traditional image measurements are separate from learned CNN features and are approximate, mask-dependent estimates.</p>
      </section>

      <EvaluationSummary
        title="Patient-disjoint validation metrics"
        metrics={validationMetrics}
        unavailableMessage="Validation metrics are not available yet. Run the validation evaluator to generate them."
      />
      <EvaluationSummary
        title="Held-out test metrics"
        metrics={metrics}
        unavailableMessage="No valid held-out evaluation report is available. The official test metadata has no labels."
      />
      {evaluationError && (
        <p className="unavailable-note" role="status">
          Latest evaluation metrics could not be refreshed: {evaluationError}
          {analysis.evaluation_metrics ? ' Showing metrics saved with this analysis instead.' : ''}
        </p>
      )}

      <section className="disclaimer-box">
        <ShieldAlert size={18} />
        <p>This experimental research prototype is not a medical diagnostic system. Never use its outputs to diagnose, rule out disease, or make treatment decisions.</p>
      </section>

      <div className="footer-actions split-links">
        <Link className="secondary-button" to="/upload">
          <RefreshCw size={16} /> Analyze another image
        </Link>
        <Link className="secondary-button" to="/history">
          <BarChart3 size={16} /> View history
        </Link>
      </div>
    </div>
  )
}
