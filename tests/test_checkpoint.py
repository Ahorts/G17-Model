"""Architecture toggles and fail-closed checkpoint loading."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from checkpoint import MODEL_REPOSITORY, MODEL_REVISION, load_checkpoint
from modeling_amfu import AMFUConfig, AMFUNet, DAG, MFEF


class CheckpointTests(unittest.TestCase):
    def test_architecture_toggles(self):
        for dag in (False, True):
            for mfef in (False, True):
                with self.subTest(dag=dag, mfef=mfef):
                    model = AMFUNet(AMFUConfig(base_channels=2, use_dag=dag, use_mfef=mfef)).eval()
                    self.assertEqual(sum(isinstance(m, DAG) for m in model.modules()), 4 if dag else 0)
                    self.assertEqual(sum(isinstance(m, MFEF) for m in model.modules()), int(mfef))
                    with torch.inference_mode():
                        logits = model(torch.zeros(1, 1, 32, 32))["logits"]
                    self.assertEqual(tuple(logits.shape), (1, 1, 32, 32))
                    self.assertTrue(torch.isfinite(logits).all())

    def test_strict_loader_rejects_incomplete_extra_and_wrong_shape(self):
        config = dict(architectures=["AMFUNet"], base_channels=32,
                      image_size=256, use_dag=True, use_mfef=False)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            model = AMFUNet(AMFUConfig(base_channels=2, use_dag=True, use_mfef=False))
            state = model.state_dict()
            cases = [dict(state), dict(state), dict(state), dict(state)]
            cases[1].pop("enc1.block.1.num_batches_tracked")
            cases[2]["unexpected.weight"] = torch.zeros(1)
            cases[3]["head.weight"] = torch.zeros(2, 2, 1, 1)
            for index, weights in enumerate(cases):
                with self.subTest(case=index), \
                        patch("checkpoint.hf_hub_download", return_value=str(path)) as download, \
                        patch("checkpoint.AMFUNet", return_value=model), \
                        patch("checkpoint.load_file", return_value=weights):
                    if index:
                        with self.assertRaises(RuntimeError):
                            load_checkpoint()
                    else:
                        self.assertIs(load_checkpoint(), model)
                    for call in download.call_args_list:
                        self.assertEqual(call.args[0], MODEL_REPOSITORY)
                        self.assertEqual(call.kwargs, dict(revision=MODEL_REVISION))

    def test_wrong_checkpoint_config_blocks_weight_download(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"use_mfef": true}', encoding="utf-8")
            with patch("checkpoint.hf_hub_download", return_value=str(path)) as download:
                with self.assertRaises(ValueError):
                    load_checkpoint()
                self.assertEqual(download.call_count, 1)
