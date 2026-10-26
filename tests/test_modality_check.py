import ast
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from modality_check import ModalityCheck, Decision, UNCERTAIN, REVISION, screen_uploads, load_modality_check


def fixture_checker():
    root = Path(__file__).parent / "fixtures" / "modalityscan"
    configs = [json.loads((root / name).read_text(encoding="utf-8")) for name in
               ("config.json", "preprocessor.json", "modality-taxonomy.json", "modality-thresholds.json")]
    return ModalityCheck(None, *configs)


class Decisions(unittest.TestCase):
    def setUp(self):
        self.check = fixture_checker()
        self.family = np.zeros(6)

    def logits(self, label):
        values = np.full(20, -8.)
        values[self.check.labels.index(label)] = 8.
        return values

    def test_us_and_other_modalities(self):
        self.assertTrue(self.check.decode(self.family, self.logits("US")).accepted)
        for label in ("CT", "MR", "DX", "OCT", "PHOTO"):
            with self.subTest(label=label):
                self.assertFalse(self.check.decode(self.family, self.logits(label)).accepted)

    def test_unsupported_slots_never_emitted(self):
        for label in self.check.tax["tiers"]["not_trained"]["types"]:
            result = self.check.decode(self.family, self.logits(label))
            self.assertFalse(result.accepted)
            self.assertNotEqual(result.label, label)

    def test_ood_and_abstention(self):
        result = self.check.decode(self.family, np.full(20, -10.))
        self.assertTrue(result.out_of_distribution)
        self.assertEqual(result.reason, UNCERTAIN)
        result = self.check.decode(self.family, np.full(20, 1.))
        self.assertEqual(result.granularity, "none")
        self.assertEqual(result.reason, UNCERTAIN)

    def test_family_and_group_are_blocked(self):
        family = np.array([0., 0., 8., 0., 0., 0.])
        result = self.check.decode(family, np.ones(20))
        self.assertEqual(result.granularity, "family")
        self.assertFalse(result.accepted)
        values = np.full(20, -8.)
        values[self.check.labels.index("CR")] = 8.
        values[self.check.labels.index("DX")] = 7.9
        result = self.check.decode(self.family, values)
        self.assertEqual(result.granularity, "group")
        self.assertFalse(result.accepted)

    def test_configured_us_threshold_and_temperature(self):
        # Construct calibrated p(US)=0.67 / 0.69, straddling configured 0.68.
        for score, expected in ((.67, False), (.69, True)):
            probs = np.full(20, (1-score)/19)
            probs[2] = score
            logits = (np.log(probs) + 10) * self.check.book["calibration"]["temperature"]["modality"]
            self.assertEqual(self.check.decode(self.family, logits).accepted, expected)

    def test_nonfinite_output_fails(self):
        with self.assertRaises(ValueError):
            self.check.decode(self.family, np.full(20, np.nan))

    def test_rgb_letterbox_and_original_preserved(self):
        image = Image.new("RGB", (100, 50), (255, 0, 128))
        original = np.asarray(image).copy()
        result = self.check.pixel_values(image)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        mean = np.array(self.check.pre["image_mean"])
        std = np.array(self.check.pre["image_std"])
        np.testing.assert_allclose(result[0, :, 0, 0], -mean/std, rtol=1e-6)
        np.testing.assert_allclose(result[0, :, 112, 112], (np.array([1,0,128/255])-mean)/std, rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(image), original)
        self.assertEqual(self.check.pixel_values(image.convert("L")).shape, result.shape)

    def test_mixed_empty_and_failed_batches(self):
        files = []
        for i in range(2):
            file = BytesIO()
            Image.new("L", (20, 10), i).save(file, format="PNG")
            file.name = f"{i}.png"
            file.seek(3)
            files.append(file)
        from unittest.mock import Mock
        checker = Mock()
        checker.classify.side_effect = [Decision("US", "modality", .9), Decision("CT", "modality", .9)]
        accepted, skipped = screen_uploads(files, checker)
        self.assertIs(accepted[0], files[0])
        self.assertEqual(files[0].tell(), 3)
        self.assertEqual(len(skipped), 1)
        checker.classify.side_effect = None
        checker.classify.return_value = Decision("CT", "modality", .9)
        self.assertEqual(screen_uploads(files, checker)[0], [])
        checker.classify.side_effect = RuntimeError("ONNX failure")
        with self.assertRaisesRegex(RuntimeError, "input check failed"):
            screen_uploads(files, checker)

    def test_new_analysis_clears_stale_results(self):
        from types import SimpleNamespace
        source = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        clear = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name == "clear_analysis")
        state = dict(raw_batch_data=["old"], skipped_files=["old"],
                     input_check_error="old", threshold=.7)
        namespace = dict(st=SimpleNamespace(session_state=state))
        exec(compile(ast.Module(body=[clear], type_ignores=[]), "app.py", "exec"), namespace)
        namespace["clear_analysis"]()
        self.assertEqual(state, dict(threshold=.7))

    def test_loading_failure_propagates(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                load_modality_check()

    def test_session_failure_and_pinned_cpu_loader(self):
        root = Path(__file__).parent / "fixtures" / "modalityscan"
        def download(repo, name, *, revision):
            self.assertEqual(repo, "vectorsense/modalityscan")
            self.assertEqual(revision, REVISION)
            return str(root / name)
        with patch("huggingface_hub.hf_hub_download", side_effect=download), \
                patch("onnxruntime.InferenceSession", side_effect=RuntimeError("invalid graph")) as session:
            with self.assertRaisesRegex(RuntimeError, "invalid graph"):
                load_modality_check()
            self.assertEqual(session.call_args.kwargs["providers"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
