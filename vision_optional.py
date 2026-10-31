"""Pretrained CLIP sky-scene *description*, not pollution detection or AQI.

A real pretrained OpenAI CLIP vision-language model ranks candidate prompts.
The output score is relative to the candidates, never a validated confidence,
health threshold, atmospheric measurement, or probability of smog.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

MODEL_ID = "openai/clip-vit-base-patch32"
SCENES = {
    "Clear sky": "an outdoor daytime photograph showing a clear blue sky with good visibility",
    "Cloudy sky": "an outdoor photograph of dense gray clouds covering the sky",
    "Possible fog": "an outdoor photograph with thick white morning fog hiding distant buildings",
    "Possible haze": "an outdoor city skyline with gray haze and low visibility in the distance",
    "Sunset or sunrise": "an outdoor landscape showing a colorful orange sunset or sunrise sky",
    "No clear sky visible": "a photograph taken indoors showing a ceiling or an object instead of a sky",
}


@lru_cache(maxsize=1)
def _encoder():
    try:
        import torch
        from transformers import CLIPModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Sky AI packages are missing. Run: py -m pip install -r requirements-vision.txt") from exc
    try:
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:
        raise RuntimeError("Could not load pretrained CLIP. First run needs internet to download its weights (~600 MB).") from exc
    model.eval()
    return model, processor


def _prepare(photo: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(photo).convert("RGB")
    image.thumbnail((1024, 1024))
    return image


def classify_sky(picture: Image.Image):
    import torch
    model, processor = _encoder()
    labels = list(SCENES)
    inputs = processor(text=list(SCENES.values()), images=_prepare(picture),
                       return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        result = model(**inputs)
        relative_scores = result.logits_per_image.softmax(dim=-1)[0].cpu().tolist()
    return sorted(zip(labels, map(float, relative_scores)), key=lambda x: -x[1])


def classify_trained(picture: Image.Image, model_path: str = "sky_prototypes.npz"):
    """Compare CLIP image embeddings with AIRFECTION labeled-photo prototypes."""
    import numpy as np
    import torch
    if not Path(model_path).is_file():
        raise FileNotFoundError("No trained sky prototype file. Run train_sky.py first.")
    model, processor = _encoder()
    inputs = processor(images=_prepare(picture), return_tensors="pt")
    with torch.inference_mode():
        vector = model.get_image_features(pixel_values=inputs["pixel_values"])
    vector = vector.detach().cpu().numpy().reshape(-1).astype("float32")
    vector /= max(np.linalg.norm(vector), 1e-9)
    with np.load(model_path, allow_pickle=False) as result:
        labels, prototypes = result["labels"], result["prototypes"]
    scores = prototypes @ vector
    # Softmax only for a ranked, relative comparison (never calibrated probability).
    relative = np.exp(12 * (scores - scores.max()))
    relative /= relative.sum()
    return sorted([(str(n), float(v)) for n, v in zip(labels, relative)], key=lambda x: -x[1])
