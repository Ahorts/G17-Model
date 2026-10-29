# Exp-25 checkpoint validation

Validated locally on 2026-10-29 with the installed app environment.

- Repository: `Ahorts/G17-AMFU-Net-Paper`
- Revision: `f2f5d8c9d95611360a5a991e74a9e8f2932b97a8`
- Experiment branch: `exp25-unet-dag-gpu-reflect-enhanced-cosine-only`
- Config: AMFUNet, base channels 32, image size 256, DAG true, MFEF false.
- Reported internal test Dice: 82.76%; this smoke test does not measure Dice
  or clinical accuracy.

All 19 automated tests pass, including architecture toggle combinations,
strict-loader failure cases (missing buffer, extra key, wrong tensor shape,
wrong config), input-check failures and cached threshold changes.

Actual safetensors loading passed explicit complete-key comparison and
`load_state_dict(strict=True)`. MFEF was not instantiated. No missing or
unexpected keys were present and no tensor shapes mismatched.

The real model and real ModalityScan checker were exercised through Streamlit
AppTest. Only the upload widget and the resource loaders were substituted to
supply local public files and already loaded real models; predictions and
classification were not mocked.

| Case | Accepted | Rejected | Finite logits | Logit range |
| --- | ---: | ---: | --- | --- |
| Single BUSI ultrasound | 1 | 0 | `[1, 1, 256, 256]` | -57.1076 to 17.2320 |
| Two BUSI ultrasounds + grayscale astronaut | 2 | 1 | `[2, 1, 256, 256]` | -57.1076 to 17.2320 |

The astronaut was identified as MR and skipped. Each run rendered ten image
panels, including masks, probability maps and overlays. Masks and overlays were
also written locally for each accepted scan. Changing the threshold from 0.50
to 0.70 preserved the probability maps exactly and caused no extra model call.

Model-input tensors matched the existing preprocessing output with zero
absolute and relative tolerance. Original, despeckled and enhanced images
matched exactly. `preprocessing.py` and `modality_check.py` are unchanged from
the previous deployment. Reflection padding was not added to inference.
Public sample provenance and limitations are documented in VALIDATION.md.

Reproduce with those samples:

```powershell
python -m unittest discover -s tests -v
python validate_checkpoint.py .validation_samples --output .validation_samples/exp25-inference.json
git diff --check
```

Generated images, model weights and local logs are excluded from the commit.
Live deployment verification is reported separately after the main merge.
