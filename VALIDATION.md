# ModalityScan input-check validation

Run on 2026-10-26 using upstream revision
`2a85d5aee29ee62bf2795cbd6e02c8ee5c248eb9`, INT8 ONNX,
ONNX Runtime 1.31.0 and `CPUExecutionProvider` only. No training occurred.
The revision was obtained from Hugging Face's model API and all four JSON
files and the model-card inference example were inspected before implementation.
The downloaded ONNX file's SHA-256 was checked against Hugging Face's LFS metadata:
`c0a5aa028f3696bbc45e11f10704ef978c7577c34b534132bd9a4d9a8ac95a71`.

## Actual inference smoke test

| Sample category | Count | Accepted as US | Gate errors |
| --- | ---: | ---: | ---: |
| Breast ultrasound | 3 | 3 | 0 false rejections |
| Photos (one RGB, two grayscale) | 3 | 0 | 0 false acceptances |
| Grayscale scanned document | 1 | 0 | 0 false acceptances |
| Binary masks | 3 | 0 | 0 false acceptances |
| Grayscale CT and MR slices | 2 | 0 | 0 false acceptances |
| Code-editor screenshot | 1 | 0 | 0 false acceptances |

Total: **0/3 false rejections, 0/10 false acceptances**. Eight of the thirteen
inputs were stored as single-channel grayscale; all eight were negatives.
The BUSI images are RGB files with grayscale ultrasound content.
These are small convenience samples, not an independent benchmark. BUSI may
overlap upstream training sources. Counts do not estimate app accuracy or
establish reliable rejection of unrelated images.

Notable failures of modality identification despite correct gate blocking:
the grayscale astronaut photo was confidently labeled MR (0.939); the camera
photo fell back to cross-sectional family, and the document, screenshot and
two masks fell back to microscopy family. The MR slice was rejected as OOD.
Passing the gate therefore requires the specific US modality, never a family.

Per-file scores, output levels, OOD flags, sizes and SHA-256 checksums are in
`validation_results.json`. Downloaded samples and weights are ignored by Git.

## Sample provenance and reproduction

- Ultrasound: first three rows of
  [SamIndia/BUSI](https://huggingface.co/datasets/SamIndia/BUSI), dataset commit
  `ceb2a2b3af5116420ec342baf688ee8393143707`, image IDs `benign (1)`,
  `benign (10)`, `benign (100)`. Retrieved via datasets-server rows endpoint,
  decoded JPEG previews saved as PNG. Corresponding mask previews were
  converted to L and thresholded at 128 to create true binary negatives.
- Photos/document: `astronaut.png`, `camera.png`, `page.png` from
  [scikit-image v0.25.2 data](https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data).
  The third photo is an L conversion of astronaut. No patient data was used.
- Medical scans: `CT_small.dcm`, `MR_small.dcm` from
  [pydicom test files](https://github.com/pydicom/pydicom/tree/main/src/pydicom/data/test_files).
  Pixel arrays were linearly min/max scaled to uint8 PNG for this test only.
  This rendering is outside the app and does not change segmentation preprocessing.
- Screenshot: first frame of
  [VS Code IntelliSense animation](https://github.com/microsoft/vscode-docs/blob/main/docs/editing/images/intellisense/intellisense.gif),
  fetched via media.githubusercontent.com and saved as RGB PNG.

After recreating these rendered images in a directory, run:

```bash
python validate_samples.py SAMPLE_DIRECTORY --output results.json
python -m unittest discover -s tests -v
git diff --check
```

Only weights/configuration are downloaded during inference; images remain local.

## Automated and integration checks

All **16 tests passed** in the installed app environment: 11 policy/loader/state
checks, four Streamlit AppTest scenarios, and one tensor-preservation test.
They cover US and other modalities, unsupported slots, calibrated thresholds,
OOD/abstention/group/family results, nonfinite outputs, mixed/all-rejected
batches, download/session/classification failures, stale-result clearing and
threshold reruns without repeating inference.

Additionally, all three actual accepted BUSI uploads were routed through the
unchanged app batch/preprocessing functions. Their file bytes, original and
enhanced arrays, and batched tensors matched the original preprocessing exactly
(tensor comparison used zero absolute and relative tolerance). The segmentation
forward pass was mocked for this check and the UI tests; the pretrained
AMFU-Net checkpoint was not downloaded or evaluated. Its architecture and
preprocessing files were left unchanged. Source comparisons also confirmed that
the existing batch inference, mask/probability derivation and overlay functions
are unchanged. Python compilation and `git diff --check` passed.

The app environment used Python 3.13, Streamlit 1.66.1, PyTorch 2.14.1,
Transformers 5.0.0, ONNX Runtime 1.31.0 and Hugging Face Hub 1.33.0. Actual
ModalityScan inference was also repeated in that environment with identical
per-file results. Streamlit emits deprecation warnings for existing
`use_container_width` calls; these do not prevent the tests from passing.

## Limits

The check recognizes ultrasound appearance. It cannot establish breast anatomy,
lesion presence or diagnostic suitability. Open-set rejection is unmeasured
upstream. There is no claim of reliable rejection of every unrelated image.
Only three benign breast ultrasound images from one source were tested; colour
Doppler, other scanners/anatomies, malignant/normal examples and a broader
adversarial/real-world negative corpus remain untested.
