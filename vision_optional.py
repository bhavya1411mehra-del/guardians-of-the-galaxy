"""AIRFECTION Sky Photo AI: visual evidence, NEVER an AQI or 'not polluted' test.

Uses pretrained CLIP to rank scene descriptions, including factory emissions.
The prompt ensemble improves coverage of categories missing in V3 but needs
independent photo validation; no predicted score is a calibrated probability.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np

MODEL_ID = "openai/clip-vit-base-patch32"

# Descriptions should refer to VISIBLE features. Avoid "this is pollution" or
# "this is not pollution", because a photograph cannot determine an AQI.
SCENE_PROMPTS = {
    "Visible industrial smoke": [
        "a factory with tall smokestacks releasing thick dark smoke plumes into the sky",
        "industrial power plant chimneys billowing dense black and gray smoke above buildings",
        "multiple industrial chimneys with long visible exhaust smoke plumes under a cloudy sky",
    ],
    "Visible fire smoke": [
        "a large plume of smoke rising from burning material outdoors",
        "thick gray smoke coming from a wildfire or fire outdoors",
        "black smoke rising from a burning fire and spreading across the sky",
    ],
    "Possible atmospheric haze": [
        "an outdoor city skyline obscured by diffuse gray atmospheric haze",
        "distant buildings partly hidden behind uniform dusty hazy air",
        "a wide hazy cityscape with low visibility and washed out gray skyline",
    ],
    "Possible fog": [
        "an outdoor street with thick white fog obscuring distant objects",
        "buildings partially hidden in dense morning mist and fog",
        "a city skyline fading into a thick layer of light gray fog",
    ],
    "Cloudy or stormy sky": [
        "a landscape with large dark storm clouds covering the sky",
        "an overcast sky filled with thick gray rain clouds above buildings",
        "dramatic dark thunderclouds over a city skyline without obvious chimney smoke",
    ],
    "Clear sky": [
        "an outdoor city landscape under a clear bright blue sky",
        "buildings with excellent visibility and a blue cloudless sky",
        "a sunny outdoor landscape with a clear blue sky",
    ],
    "Sunrise or sunset": [
        "an outdoor landscape beneath a colorful orange and pink sunset",
        "city buildings photographed under a golden sunrise sky",
        "the sun setting at the horizon with warm red and orange sky colors",
    ],
    "Unrelated or unclear scene": [
        "an indoor room with no outdoor sky visible",
        "a closeup photo of an object where the sky cannot be seen",
        "a photograph of a person indoors with no outdoor atmosphere visible",
    ],
}
SMOKE_CLASSES = frozenset({"Visible industrial smoke", "Visible fire smoke"})


@lru_cache(maxsize=1)
def _encoder():
    try:
        from transformers import CLIPModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Missing CLIP packages. Install requirements-vision.txt in the app's Python environment.") from exc
    try:
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:
        raise RuntimeError("Pretrained CLIP weights could not load. Check internet, memory and hosting logs.") from exc
    model.eval()
    return model, processor


def _prepare(photo: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(photo).convert("RGB")
    image.thumbnail((1024, 1024))
    return image


def grouped_scene_scores(prompt_logits, prompt_groups=SCENE_PROMPTS):
    """Average CLIP prompt logits per category, then normalize across classes.

    This avoids giving classes with more prompt variants an automatic advantage.
    Values are relative similarities, NOT probabilities that air is polluted.
    """
    values = np.asarray(prompt_logits, dtype="float64").reshape(-1)
    sizes = [len(prompts) for prompts in prompt_groups.values()]
    if len(values) != sum(sizes) or any(s < 1 for s in sizes):
        raise ValueError("One CLIP score is required for each text prompt")
    summaries, i = [], 0
    for n in sizes:
        summaries.append(float(values[i:i+n].mean()))
        i += n
    summaries = np.asarray(summaries)
    relative = np.exp(summaries - summaries.max())
    relative = relative / relative.sum()
    return sorted(zip(prompt_groups.keys(), relative.tolist()), key=lambda item: -item[1])


def classify_sky(picture: Image.Image):
    """Run real pretrained CLIP over multiple descriptions for each scene."""
    import torch
    model, processor = _encoder()
    texts = [p for prompts in SCENE_PROMPTS.values() for p in prompts]
    inputs = processor(text=texts, images=_prepare(picture),
                       return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0].detach().float().cpu().numpy()
    return grouped_scene_scores(logits)


def classify_trained(picture: Image.Image, model_path: str = "sky_prototypes.npz"):
    """Compare pretrained CLIP embeddings with *your* labeled sky photo prototypes."""
    import torch
    if not Path(model_path).is_file():
        raise FileNotFoundError("No trained sky prototype file. Train with train_sky.py first.")
    model, processor = _encoder()
    inputs = processor(images=_prepare(picture), return_tensors="pt")
    with torch.inference_mode():
        vector = model.get_image_features(pixel_values=inputs["pixel_values"])
    vector = vector.detach().cpu().numpy().reshape(-1).astype("float32")
    vector /= max(np.linalg.norm(vector), 1e-9)
    with np.load(model_path, allow_pickle=False) as result:
        labels, prototypes = result["labels"], result["prototypes"]
    scores = prototypes @ vector
    relative = np.exp(12 * (scores - scores.max()))
    relative /= relative.sum()
    return sorted([(str(n), float(v)) for n, v in zip(labels, relative)], key=lambda x: -x[1])


def interpretation(scored):
    """Never claim that a picture proves clean air, regardless of its top label."""
    if not scored:
        return "No analysis is available. Air quality is unknown."
    top = scored[0][0].lower()
    if "smoke" in top:
        return "Visible smoke is a possible match. This is NOT an AQI or PM2.5 measurement."
    if "haze" in top:
        return "Possible haze is a visual match. It might also be fog or cloud; no air quality measurement is possible."
    return ("The top scene description does not establish that the air is clean. "
            "Smoke can be missed or confused with clouds. Air quality remains unknown.")
