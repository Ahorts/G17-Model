"""Actual pinned-checkpoint and Streamlit flow smoke test on local public samples.

python validate_checkpoint.py SAMPLE_DIRECTORY --output REPORT.json
Expects busi_0_image.png, busi_1_image.png and astronaut_gray.png, as documented
in VALIDATION.md. Writes masks and overlays beside the report; keep these local.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch

from checkpoint import MODEL_REPOSITORY, MODEL_REVISION, load_checkpoint
from modality_check import load_modality_check
from modeling_amfu import MFEF
from preprocessing import preprocess_ultrasound
import ast


def overlay_function():
    # Execute the actual app helper without running its top-level UI twice.
    import cv2
    source = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == "create_overlay")
    namespace = dict(cv2=cv2, np=np)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace["create_overlay"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_checkpoint()
    assert not any(isinstance(module, MFEF) for module in model.modules())
    checker = load_modality_check()
    calls = []

    def check_forward(module, inputs, kwargs, output):
        tensor = kwargs["pixel_values"]
        logits = output["logits"]
        assert tuple(logits.shape) == (tensor.shape[0], 1, 256, 256)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(tensor, torch.cat(expected_tensors), rtol=0, atol=0)
        calls.append(dict(shape=list(logits.shape), minimum=logits.min().item(),
                          maximum=logits.max().item()))

    hook = model.register_forward_hook(check_forward, with_kwargs=True)
    rows = []
    create_overlay = overlay_function()
    for label, names in [
        ("single", ["busi_0_image.png"]),
        ("mixed", ["busi_0_image.png", "astronaut_gray.png", "busi_1_image.png"]),
    ]:
        st.cache_resource.clear()
        files = []
        expected_tensors = []
        expected_images = []
        for name in names:
            file = BytesIO((args.directory / name).read_bytes())
            file.name = name
            files.append(file)
            if name.startswith("busi"):
                expected = preprocess_ultrasound(Image.open(BytesIO(file.getvalue())))
                expected_tensors.append(expected["tensor"])
                expected_images.append(expected)
        with patch("streamlit.file_uploader", return_value=files), \
                patch("modality_check.load_modality_check", return_value=checker), \
                patch("checkpoint.load_checkpoint", return_value=model):
            app = AppTest.from_file("app.py", default_timeout=120).run()
            app.button[0].click().run()
            assert not app.exception, app.exception
            assert not app.error, app.error
            raw = app.session_state["raw_batch_data"]
            assert len(raw) == len(expected_tensors)
            skipped = app.session_state["skipped_files"]
            assert len(skipped) == len(names) - len(raw)
            assert len(calls) == len(rows) + 1
            first_probabilities = [item["probability"].copy() for item in raw]
            rendered_images = len(app.get("image"))
            for index, (item, expected) in enumerate(zip(raw, expected_images)):
                for key in ("original", "enhanced", "despeckled"):
                    np.testing.assert_array_equal(item[key], expected[key])
                assert np.isfinite(item["probability"]).all()
                mask = (item["probability"] >= .5).astype(np.uint8) * 255
                Image.fromarray(mask).save(args.output.parent / f"{label}-{index}-mask.png")
                overlay = create_overlay(item["original"], mask)
                assert overlay.shape == (*mask.shape, 3)
                Image.fromarray(overlay).save(args.output.parent / f"{label}-{index}-overlay.png")
            assert rendered_images >= 4, [(node.type, type(node).__name__) for node in app]
            app.slider[0].set_value(.7).run()
            assert not app.exception
            assert len(calls) == len(rows) + 1, "Threshold repeated model inference"
            for previous, item in zip(first_probabilities, app.session_state["raw_batch_data"]):
                np.testing.assert_array_equal(previous, item["probability"])
            rows.append(dict(case=label, accepted=len(raw), skipped=skipped,
                             threshold_reuses_predictions=True, rendered_images=rendered_images,
                             logits=calls[-1]))
            print(json.dumps(rows[-1]), flush=True)
    hook.remove()
    args.output.write_text(json.dumps(dict(repository=MODEL_REPOSITORY,
        revision=MODEL_REVISION, strict_load=True, mfef_instantiated=False,
        cases=rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
