"""Load the selected experiment without permissive Transformers key handling."""
import json

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from modeling_amfu import AMFUConfig, AMFUNet


MODEL_REPOSITORY = "Ahorts/G17-AMFU-Net-Paper"
MODEL_BRANCH = "exp25-unet-dag-gpu-reflect-enhanced-cosine-only"
MODEL_REVISION = "f2f5d8c9d95611360a5a991e74a9e8f2932b97a8"
MODEL_LABEL = "Exp-25: U-Net + DAG, SRAD + CLAHE"


def load_checkpoint():
    """Return a CPU eval model only after every safetensors key loads strictly."""
    config_path = hf_hub_download(
        MODEL_REPOSITORY, "config.json", revision=MODEL_REVISION
    )
    with open(config_path, encoding="utf-8") as handle:
        values = json.load(handle)
    expected = dict(architectures=["AMFUNet"], base_channels=32,
                    image_size=256, use_dag=True, use_mfef=False)
    for key, value in expected.items():
        if key not in values or type(values[key]) is not type(value) or values[key] != value:
            raise ValueError(f"Unexpected Exp-25 config {key}: {values.get(key)!r}")

    weights_path = hf_hub_download(
        MODEL_REPOSITORY, "model.safetensors", revision=MODEL_REVISION
    )
    model = AMFUNet(AMFUConfig(**values))
    # Check all keys explicitly: PyTorch permits legacy checkpoints to omit
    # BatchNorm num_batches_tracked buffers even with strict=True.
    weights = load_file(weights_path, device="cpu")
    expected_keys = set(model.state_dict())
    missing = sorted(expected_keys - weights.keys())
    unexpected = sorted(weights.keys() - expected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model
