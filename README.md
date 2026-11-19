# DermaVision AI

Research prototype for **Augmented Reality for Early Skin Disease Assessment Using Computer Vision**. The existing React frontend and FastAPI backend are retained.

## Dataset inspection and storage

The pipeline reads metadata and image bytes directly from `archive.zip`; it never extracts the dataset or changes the ZIP. The verified archive contains:

- 33,126 labeled training images and 10,982 test images, all matched to their metadata IDs.
- Training labels in `target`: `0` = non-melanoma (32,542 rows), `1` = melanoma (584 rows).
- The official test CSV has no target column, so it cannot be used to calculate classification metrics.
- No segmentation-mask files were found. OpenCV segmentation results are approximate and are not ground-truth annotations.

The archive is about 31.99 GB compressed (about 30.97 GiB uncompressed). Images are read lazily from the ZIP. The ResNet18 checkpoint is typically tens of megabytes; PyTorch/torchvision installation requires additional space, and the exact package footprint depends on the selected wheels. Check available space before installing or training. The model is trained from ZIP streams; no multi-gigabyte extracted image copy is needed.

The current CPU-only machine has 7.84 GiB visible RAM (8 GiB installed), with about 0.75 GiB free and about 9.06 GiB free disk space at the latest check; available RAM fluctuates. Disk space is adequate for model artifacts without extracting the archive, but the current free RAM is too tight for CNN training. Close other applications and confirm several GiB of RAM are free before training. The full run processes 20,423 training and 6,043 validation images for each of 8 epochs, in addition to the separately reserved 6,660-image test fold. Conservative defaults are a batch size of 4, zero data-loader workers, and two CPU threads. On the 4-core Intel i5 CPU, expect many hours and potentially more than a day; no full-dataset runtime benchmark has been performed. ResNet18 weights are tens of megabytes, so model artifacts require little disk relative to the archive. Do not treat the small smoke run as evidence of generalization.

Run commands from PowerShell:

```powershell
cd C:\Users\KRISTI\Desktop\MINOR\backend
.\.venv\Scripts\Activate.ps1
python scripts\inspect_dataset.py
```

The inspection report is written to `backend\data\dataset_inspection.json`.

## CNN training and evaluation

Install the backend requirements after checking free space:

```powershell
cd C:\Users\KRISTI\Desktop\MINOR\backend
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The default training run uses torchvision's pretrained ResNet18 weights. On its first run, torchvision downloads the official ImageNet checkpoint (about 45 MB); subsequent runs use its local cache. To avoid downloading pretrained weights, pass `--no-pretrained` for random initialization instead.

Run a small patient-disjoint smoke test first:

```powershell
python scripts\train_model.py --smoke-test --no-pretrained
```

Smoke-test files are saved separately and are not activated by the application. Images are read lazily from ZIP streams without extraction; duplicate image IDs and Pillow decompression-bomb warnings fail explicitly. Full training uses the complete labeled training data with aspect-preserving centered resize with ImageNet-mean padding, ImageNet RGB normalization, train-only augmentation, class-weighted loss, and patient-disjoint train/validation/test folds. It prints validation loss and validation metrics per epoch, atomically saves the best model and preprocessing configuration, and maintains a separate atomically replaced epoch checkpoint (including optimizer and random-number-generator state) for resuming interrupted runs. The production checkpoint is only updated when validation loss improves. The run reserves a separate patient-level test fold:

```powershell
python scripts\train_model.py --epochs 8 --batch-size 4 --num-workers 0 --cpu-threads 2
```

If training is interrupted, resume from the latest completed epoch using the same data split and training settings:

```powershell
python scripts\train_model.py --epochs 8 --batch-size 4 --num-workers 0 --cpu-threads 2 --resume
```

Wait for training to finish successfully and print `training_complete` with the production checkpoint and preprocessing configuration saved before running evaluation as a separate command:

```powershell
python scripts\evaluate_model.py --batch-size 4 --cpu-threads 2
```

The evaluator reports accuracy, precision, melanoma recall/sensitivity, specificity, F1, ROC-AUC, and a confusion matrix on the untouched patient-level test fold reserved from the labeled training metadata. The decision threshold is fixed at 0.5; it is not tuned on the held-out test fold. It never uses the unlabeled official test set for metrics. Evaluation results appear only after the evaluation script runs; no placeholder metrics are shown as CNN results.

Model artifacts are saved under `backend\model\`; evaluation output is saved under `backend\results\cnn_evaluation_metrics.json`.
The pre-existing `backend\results\evaluation_metrics.json` and `backend\model\baseline_model.joblib` belong to the earlier 16×16 logistic-regression prototype; the CNN API and frontend do not present those values as CNN output.

To reproduce the production checkpoint's metrics on the patient-disjoint labeled validation fold without changing the checkpoint or held-out test report:

```powershell
python scripts\evaluate_validation.py --batch-size 8 --cpu-threads 2
```

Validation metrics and per-image predictions are written separately to `backend\results\validation_evaluation\`. The report includes class-wise melanoma-score distributions, label and patient-split checks, checkpoint identity, and the fixed 0.50-threshold metrics. It also reports exploratory thresholds for 90% and 95% validation sensitivity to show the sensitivity/specificity tradeoff; these thresholds are not applied to production predictions or presented as clinical recommendations. The original training run did not save per-epoch training-loss history; its best validation loss is recorded in the model configuration.

To run the CV and feature-extraction demonstration on 5–10 real training images without extracting the ZIP or loading a classifier:

```powershell
python scripts\create_segmentation_demo.py --count 8
```

The script writes one five-panel original/preprocessed/mask/segmented/boundary comparison per image and `sample_features.csv` under `backend\results\cv_pipeline_verification\`. The CSV marks failed segmentations/features unavailable instead of inventing values. These masks are demonstrations, not verified annotations.

## Computer vision and API

Uploaded images follow this order: RGB decode and aspect-preserving resize with centered ImageNet-mean padding/normalization → approximate GrabCut segmentation with a Lab border-colour contrast fallback, morphological cleanup, and rejection of empty, near-empty, near-full, or border-touching masks → mask-dependent traditional feature extraction → CNN classification of the preprocessed RGB image (not the masked image). The gray bars in the preprocessed input are intentional ImageNet-mean padding; training and inference use the same 224×224 resize and normalization. Segmentation logs report the selected method, input and mask ranges/shapes, foreground count/percentage, contour count, and failure reason. Failed masks are not returned as black mask images, and lesion-specific features remain unavailable. Optional OpenCV hair reduction is available in the preprocessing module but disabled by default to keep training and inference consistent.

The upload endpoint accepts JPEG/PNG content, rejects images smaller than 64×64 or larger than 40 megapixels, and rejects near-uniform or severely out-of-focus inputs using conservative checks on a 256×256 thumbnail. These checks only flag inputs with insufficient visible detail; they do not establish that an image is suitable for diagnosis. The classifier was trained on ISIC2020 dermoscopic images; camera/modality metadata is not available in the archive, and ordinary phone photographs have not been independently validated.

If no trained full-model checkpoint exists, the API returns preprocessing, segmentation, and available features with an explicit unavailable prediction status. It does not use the legacy demonstration model to fabricate a prediction.

Run the backend:

```powershell
cd C:\Users\KRISTI\Desktop\MINOR\backend
.\.venv\Scripts\Activate.ps1
python -m app.main
```

Health: <http://localhost:8000/api/health>  
Dataset summary: <http://localhost:8000/api/dataset/summary>  
CNN metrics: <http://localhost:8000/api/evaluation/metrics>

## Frontend

```powershell
cd C:\Users\KRISTI\Desktop\MINOR\frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Upload an image to see its original and preprocessed images, approximate mask and boundary, traditional features, prediction status/probabilities, and available evaluation metrics.

## Live camera and 2D AR preview

Open **AR Preview** in the React application. The page automatically requests camera permission and prefers the rear-facing camera:

1. Allow browser camera permission.
2. Keep the lesion inside the circular scanning frame with bright, even lighting and no filters or digital zoom.
3. The page samples a new JPEG frame approximately every 1.5 seconds. Only one backend request is active at a time.
4. Each frame is sent to the existing `/api/analyze` endpoint with `source=camera&realtime=true`; realtime frames are not saved to SQLite history.
5. Returned segmentation contours, bounding rectangles and CNN probabilities are drawn over the live video. The last valid overlay remains visible if a later frame fails.
6. Use **Stop Scanning**, **Switch Camera**, and the boundary/label/panel controls to manage the live experience.
7. The AR Preview page is a genuine 2D HTML/CSS canvas image overlay. It is not world-tracked, depth-aware, or full 3D augmented reality.

Browser camera access generally requires HTTPS or `localhost`. When testing on a phone, run the frontend/backend over a secure HTTPS tunnel or development host that is reachable by the phone. Do not deploy or publish a camera-enabled build without explicit approval.

The held-out report is a serious limitation: the current model has 0% melanoma sensitivity at the fixed 0.50 threshold and missed all 122 melanoma-positive cases in that fold. Ordinary smartphone-camera images have not been independently validated. Camera or upload outputs must not be treated as diagnoses or used to delay professional medical assessment.

## Non-technical user guide

- Start the backend and frontend with the PowerShell commands below.
- Open the frontend in a supported browser.
- Use **Upload** to choose a JPG or PNG manually, or use **AR Preview** for continuous live-camera analysis.
- Review preprocessing, the approximate segmentation, extracted features and model output on the **Results** page.
- Open **History** to review or delete saved manual analyses. Realtime camera frames are not saved to history.
- Seek qualified professional assessment for any real skin concern. The application is for research and education only.

## Research and medical disclaimer

This prototype and its approximate segmentation are for research and education only. Outputs are not clinical diagnoses and must not be used to diagnose, rule out, or treat a condition. No segmentation accuracy is reported without suitable reference masks.
