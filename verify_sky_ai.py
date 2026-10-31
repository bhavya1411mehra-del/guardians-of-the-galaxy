"""Smoke test pretrained CLIP without claiming accuracy; requires first-time download."""
from PIL import Image
from vision_optional import classify_sky

if __name__ == "__main__":
    print("Loading real pretrained CLIP — download occurs on first run...")
    # An empty image checks that inference works; the returned label is meaningless.
    image = Image.new("RGB", (240, 240), (125, 150, 170))
    labels = classify_sky(image)
    assert labels and all(isinstance(score, float) for _, score in labels)
    print("SUCCESS: CLIP model loaded and inference executed.")
    print("This synthetic test image does NOT validate haze/fog classification.")
