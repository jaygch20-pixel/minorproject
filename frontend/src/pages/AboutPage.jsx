import { ShieldAlert } from 'lucide-react'

export default function AboutPage() {
  return (
    <div className="page-stack narrow">
      <section className="panel about-panel">
        <span className="eyebrow">About the project</span>
        <h1>DermaVision AI — Skin Lesion Research Assistant</h1>
        <p>
          This project explores how computer vision and machine learning can assist with early skin disease assessment,
          especially melanoma screening workflows. The aim is to support academic research and educational learning,
          not to provide clinical diagnosis.
        </p>

        <h3>Research objective</h3>
        <p>
          The prototype is focused on evaluating whether an uploaded lesion image can be passed through a machine learning
          pipeline and displayed in a clean research interface with results, explanations, and a 2D camera-frame overlay.
        </p>

        <h3>Dataset and method</h3>
        <p>
          The project inspects the ISIC 2020 archive directly without modifying the original dataset. The metadata includes a
          melanoma target column where 0 represents non-melanoma and 1 represents melanoma. The dataset is read from the
          archive in a streaming-safe way, and the prototype keeps image data out of RAM unless necessary for a single upload.
        </p>

        <h3>Current implementation status</h3>
        <p>
          The frontend, backend, upload flow, results page, and history system are implemented and connected to the API.
          A production ResNet18 checkpoint trained on the ISIC2020 archive is available. It uses ImageNet-compatible
          preprocessing and converts two logits into non-melanoma/melanoma probabilities. Approximate OpenCV segmentation
          and traditional feature extraction remain separate from CNN classification.
        </p>

        <h3>Live 2D camera visualization</h3>
        <p>
          The AR Preview page uses the browser MediaDevices API for continuous live-camera analysis and sends sampled JPEG frames
          every 1.5 seconds through the existing FastAPI analysis endpoint. Its visualization is a 2D image overlay: it does not use spatial
          world tracking, depth sensing, or 3D augmented reality.
        </p>

        <h3>Known model limitations</h3>
        <p>
          At the fixed 0.50 held-out threshold, the current model reports 0% melanoma sensitivity: it detected 0 of 122
          melanoma-positive cases in the held-out test report and 0 of 135 in the validation report. A non-melanoma result
          does not rule out melanoma. The model was trained on dermoscopic ISIC2020 images, and ordinary smartphone
          photographs have not been independently validated.
        </p>

        <h3>Ethical limitations</h3>
        <div className="warning-box inline-warning">
          <ShieldAlert size={16} />
          This prototype must not be used to diagnose or rule out melanoma. It is an experimental academic tool only.
        </div>
      </section>
    </div>
  )
}
