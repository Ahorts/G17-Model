"""Pinned, CPU-only ModalityScan inference; no segmentation pixel transformations."""
from dataclasses import dataclass
import json

import numpy as np
from PIL import Image

REPOSITORY = "vectorsense/modalityscan"
REVISION = "2a85d5aee29ee62bf2795cbd6e02c8ee5c248eb9"
UNCERTAIN = "Could not confidently identify this as an ultrasound image."


@dataclass(frozen=True)
class Decision:
    label: str
    granularity: str
    score: float
    out_of_distribution: bool = False

    @property
    def accepted(self):
        return (self.label == "US" and self.granularity == "modality"
                and not self.out_of_distribution)

    @property
    def reason(self):
        if self.accepted:
            return "Ultrasound appearance identified."
        if self.granularity == "modality" and not self.out_of_distribution:
            return f"Identified as {self.label}, rather than US."
        return UNCERTAIN


class ModalityCheck:
    def __init__(self, session, config, preprocessor, taxonomy, thresholds):
        self.session = session
        self.config = config
        self.pre = preprocessor
        self.tax = taxonomy
        self.book = thresholds
        self.labels = taxonomy["levels"]["modality"]["index"]
        self.families = taxonomy["levels"]["family"]["index"]
        self.tiers = {label: tier for tier in
                      ("model_trained", "experimental", "not_trained")
                      for label in taxonomy["tiers"][tier]["types"]}
        if (config["taxonomy_version"] != taxonomy["taxonomy_version"]
                or thresholds["taxonomy_version"] != taxonomy["taxonomy_version"]
                or preprocessor["geometry"]["mode"] != "letterbox"
                or preprocessor["geometry"]["interpolation"] != "bilinear"
                or preprocessor["layout"] != "NCHW"
                or preprocessor["dtype"] != "float32"
                or preprocessor["channels"] != 3
                or preprocessor["image_size"] != config["image_size"]
                or set(self.labels) != set(self.tiers)
                or self.tiers.get("US") != "model_trained"):
            raise ValueError("Incompatible ModalityScan configuration.")

    def pixel_values(self, image):
        # Follow the model-card serving example: resize longest side, centre on black.
        image = image.convert("RGB")
        size = self.pre["image_size"]
        scale = size / max(image.size)
        small = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))),
                             Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size),
                           (self.pre["geometry"]["pad_value"],) * 3)
        canvas.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
        array = np.asarray(canvas, np.float32) * self.pre["rescale_factor"]
        mean = np.asarray(self.pre["image_mean"], np.float32)[:, None, None]
        std = np.asarray(self.pre["image_std"], np.float32)[:, None, None]
        return np.ascontiguousarray(((array.transpose(2, 0, 1) - mean) / std)[None])

    def decode(self, family_logits, modality_logits):
        book = self.book
        policy = book["default_policy"]
        limits = book["policies"][policy]
        temperature = book["calibration"]["temperature"]
        probabilities = []
        energy = None
        for values, level, width in ((family_logits, "family", len(self.families)),
                                      (modality_logits, "modality", len(self.labels))):
            x = np.asarray(values, dtype=np.float64)
            if x.shape != (width,) or not np.isfinite(x).all():
                raise ValueError(f"Invalid ModalityScan {level} output.")
            x = x / temperature[level]
            logsumexp = float(x.max() + np.log(np.exp(x - x.max()).sum()))
            probabilities.append(np.exp(x - logsumexp))
            if level == "modality":
                energy = -logsumexp
        p_fam, p_mod = probabilities
        if book["ood"]["enabled"] and energy > book["ood"]["energy_threshold"][policy]:
            return Decision(book["ood"]["fallback_label"], "none", 0., True)
        # Unsupported slots remain in the softmax denominator, but are never emitted.
        best = next(i for i in np.argsort(-p_mod)
                    if self.tiers[self.labels[i]] != "not_trained")
        label, score = self.labels[best], float(p_mod[best])
        floor = book["per_class_overrides"].get(policy, {}).get(label, limits["modality"])
        if score >= floor:
            return Decision(label, "modality", score)
        if book["cascade"]["enabled"]:
            for group, entry in self.tax["collapse_groups"]["groups"].items():
                if label in entry["members"]:
                    mass = sum(p_mod[self.labels.index(member)] for member in entry["members"]
                               if self.tiers[member] != "not_trained")
                    if mass >= limits["group"]:
                        return Decision(group, "group", float(mass))
            best_family = int(p_fam.argmax())
            if p_fam[best_family] >= limits["family"]:
                return Decision(self.families[best_family], "family", float(p_fam[best_family]))
        return Decision(book["abstention"]["modality_indeterminate"], "none", score)

    def classify(self, image):
        family, modality = self.session.run(
            ["family_logits", "modality_logits"],
            {self.pre["input_name"]: self.pixel_values(image)})
        return self.decode(family[0], modality[0])


def load_modality_check():
    # Imports and downloads occur inside the guarded UI call; failures block analysis.
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    def download(name):
        return hf_hub_download(REPOSITORY, name, revision=REVISION)

    configs = []
    for name in ("config.json", "preprocessor.json", "modality-taxonomy.json",
                 "modality-thresholds.json"):
        with open(download(name), encoding="utf-8") as file:
            configs.append(json.load(file))
    session = ort.InferenceSession(download("model.int8.onnx"),
                                   providers=["CPUExecutionProvider"])
    return ModalityCheck(session, *configs)


def screen_uploads(files, checker):
    """Classify originals; preserve upload bytes and cursor for segmentation."""
    from io import BytesIO
    accepted, skipped = [], []
    for file in files:
        try:
            with Image.open(BytesIO(file.getvalue())) as image:
                image.load()
                decision = checker.classify(image)
        except Exception as exc:
            raise RuntimeError(f"Ultrasound input check failed for {file.name}: {exc}") from exc
        if decision.accepted:
            accepted.append(file)
        else:
            skipped.append({"Filename": file.name, "Reason": decision.reason})
    return accepted, skipped
