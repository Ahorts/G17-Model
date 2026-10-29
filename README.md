# G17 AMFU-Net Streamlit Demo

This Streamlit application demonstrates breast-lesion segmentation with the
selected Ahortz model: **Exp-25: U-Net + DAG, SRAD + CLAHE**.
Reported internal test Dice: **82.76%**, not clinical accuracy.
It applies SRAD despeckling and CLAHE
enhancement before generating a probability map, binary mask and overlay.

Before any enhancement, the app runs the pretrained
[ModalityScan INT8 ONNX model](https://huggingface.co/vectorsense/modalityscan)
locally with ONNX Runtime's `CPUExecutionProvider`. No classifier is trained.
Weights and all four configuration files are pinned to verified commit
`2a85d5aee29ee62bf2795cbd6e02c8ee5c248eb9` and cached together using
`st.cache_resource`; Hugging Face also caches downloads on disk.

The check uses RGB, bilinear aspect-preserving letterboxing to 224 pixels,
black padding, and configured ImageNet normalization. It applies configured
temperature scaling, energy rejection, supported-class filtering and the
modality/group/family abstention cascade. The supplied default `balanced`
policy includes a US override of 0.68; this is separate from the adjustable
segmentation pixel threshold. Only a specific `US` modality result passes.
Groups, families, unsupported classes, abstentions and OOD results cannot pass.
Accepted uploads enter the original SRAD/CLAHE/AMFU-Net pipeline unchanged.

Mixed batches process accepted files and list rejected filenames with reasons.
An entirely rejected batch produces no segmentation. Any input-check loading,
decoding or classification error blocks the entire run, including otherwise
accepted files. New runs and upload changes clear previous results.

This check recognizes ultrasound appearance; it does not establish breast
anatomy, lesion presence or diagnostic suitability. The upstream model card
reports US recall of 95.58% and precision of 86.10% on its evaluation corpus.
These are not accuracy measurements for this app. Open-set rejection is
explicitly unmeasured; unrelated images may pass and ultrasound may be rejected.
Screenshots of ultrasound can also resemble ultrasound. See
[validation notes](VALIDATION.md) for local sample counts and limitations.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and add these files to its root.
2. Go to https://share.streamlit.io and select **Create app**.
3. Choose the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy the app.

The config and `model.safetensors` are downloaded from
`Ahorts/G17-AMFU-Net-Paper`, pinned to revision
`f2f5d8c9d95611360a5a991e74a9e8f2932b97a8`.
Experiment branch: `exp25-unet-dag-gpu-reflect-enhanced-cosine-only`.
This is a Git revision, not a subfolder. The verified config is `AMFUNet`,
base channels 32, image size 256, DAG enabled and MFEF disabled.
MFEF is neither instantiated nor applied for this checkpoint. Architecture
toggles match [Ahortz's training implementation](https://github.com/aee4/AMFU-Net-G17-/blob/main/training/training_marimo.py).
Weights load with `load_state_dict(strict=True)`; missing, unexpected or
shape-mismatched keys fail the run. There is no partial-weight fallback.
Inference preserves SRAD + CLAHE, area resize and `(x / 255 - 0.5) / 0.5`
normalization. Reflection padding was a training augmentation and is not
applied during inference.
The model is cached after loading, so it is not downloaded for every prediction.
Both models are loaded lazily on analysis; the segmentation model is loaded
only when the input check accepts at least one file. Upload pixels are processed
locally by the app; model downloads require network access on the first run.

Research demonstration only. This model assumes a lesion is present and is not
a clinical diagnostic tool.
Do not upload identifiable patient data.

## Checks

```bash
python -m unittest discover -s tests -v
python validate_checkpoint.py SAMPLE_DIRECTORY --output LOCAL_REPORT.json
```

Test configuration fixtures are unmodified Apache-2.0 ModalityScan JSON files
from the pinned revision. They allow policy tests without model downloads.
Runtime dependencies are still required for loader and app integration tests.
The checkpoint smoke test uses the public samples documented in VALIDATION.md;
keep its generated report, masks and overlays outside tracked source files.
