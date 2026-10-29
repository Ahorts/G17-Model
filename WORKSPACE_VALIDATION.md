# Research workspace validation

Validated on 2026-10-29 with Python 3.13, Streamlit 1.66.1 and the existing
model dependencies. Hosting remains Streamlit Community Cloud, using `app.py`
on `main`. No computation dependencies or hosting settings changed.

## Automated and numerical checks

All **24 unit/integration tests pass**. The five additional lifecycle tests cover
actual stage boundaries, cached model loads, duplicate-filename selection,
threshold reruns, all-rejected batches, model-download and forward failures,
retry after a failed load and terminal states that cannot reactivate.
Upload/reset checks now also cover the saved processing state.

Actual inference used `Ahorts/G17-AMFU-Net-Paper` revision
`f2f5d8c9d95611360a5a991e74a9e8f2932b97a8`. All 88 state keys load strictly;
there are no missing, unexpected or mismatched keys, and no MFEF module.

- A single public BUSI ultrasound produced finite `[1, 1, 256, 256]` logits.
- A mixed batch accepted two BUSI ultrasounds, rejected a grayscale astronaut
  photo as MR, and produced finite `[2, 1, 256, 256]` logits.
- Logits ranged from -57.1076317 to 17.2319527 in these samples.
- All result views rendered. Threshold changes reused the predictions.
- A direct comparison against the pre-redesign app at `f6920f1` found
  bit-for-bit identical original, SRAD, enhanced and probability arrays for two
  real scans. Masks and overlays were also bit-for-bit identical at thresholds
  0.30, 0.50 and 0.70; numeric summary values were identical.
- `preprocessing.py`, `modality_check.py`, `modeling_amfu.py`, `checkpoint.py`
  and `requirements.txt` are unchanged.

Public sample provenance and model limitations remain in VALIDATION.md and
EXP25_VALIDATION.md. This is a functional smoke test, not an accuracy benchmark.
82.76% is the reported internal test Dice, not clinical accuracy.

## Browser verification

Chromium screenshots and layout checks were performed at **1440, 768 and
390 pixels** for empty, completed, mixed, all-rejected, model-loading failure
and inference failure states. Native layout reflow was allowed to settle before
checking overflow. Neither the document nor the main workspace overflowed
horizontally. Original/overlay comparisons were inspected at all three widths.

The animation was observed during a real six-image batched CPU forward pass,
without any artificial server delay. Both sampled frames remained in the
`segmenting`/`running` state while the sweep transform changed from
`translateY(-36.4318px)` to `translateY(-29.6502px)`. An active-stage screenshot
and a browser recording were saved locally. The waveform and indeterminate bar
are explicitly labelled as illustrative, not live model output.

Reduced-motion mode was checked during a real forward pass: animation name
was `none`, with operation text and the stepper still visible. Keyboard arrow
keys adjusted the threshold and the slider had a visible solid 2px focus outline.
No page-wide entrance animation runs on threshold changes.

Failure previews used a separate ignored local harness that deliberately raises
download/forward exceptions. No failure switches or test delays exist in the
deployed app. Failure and rejection states stop animation; successful runs
replace the large active card with a compact, expandable execution history.

Screenshots, recordings, sample uploads, weight files and generated reports
remain local and are excluded from Git. These checks emulate viewport sizes;
physical mobile devices and every browser/assistive-technology combination
have not been tested. Live deployment verification is reported after merging.

## Reproduce

```powershell
python -m unittest discover -s tests -v
python validate_checkpoint.py .validation_samples --output .validation_samples/redesign-inference.json
streamlit run app.py
git diff --check
```
