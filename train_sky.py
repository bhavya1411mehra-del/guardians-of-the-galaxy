"""Optional supervised sky-label training on your own labeled photographs.

Fine-tunes *class prototypes*, not the whole CLIP neural network. Uses pretrained
CLIP image embeddings and mean class centroids. Evaluates on held-out photos.

Folder structure:
 sky_photos/clear/*.jpg; sky_photos/cloudy/*.jpg; sky_photos/fog/*.jpg; sky_photos/haze/*.jpg
Run: python train_sky.py --data sky_photos --output sky_prototypes.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def image_embeddings(paths: list[Path]):
    """Pretrained CLIP is downloaded from Hugging Face on first use."""
    import torch
    from transformers import CLIPModel, AutoProcessor
    from PIL import Image, ImageOps
    print('Loading pretrained CLIP image encoder (first run downloads model weights)…')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model.eval()
    vectors = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as img:
                clean = ImageOps.exif_transpose(img).convert('RGB')
                inputs = processor(images=clean, return_tensors='pt')
            vec = model.get_image_features(pixel_values=inputs['pixel_values'])
            vec = vec.detach().cpu().numpy().reshape(-1).astype('float32')
            vec /= max(np.linalg.norm(vec), 1e-9)
            vectors.append(vec)
    return np.vstack(vectors)


def train(data: str, output: str, seed: int = 42):
    root = Path(data)
    extensions = {'.jpg','.jpeg','.png','.webp'}
    label_folders = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if len(label_folders) < 2:
        raise ValueError('Create at least two labeled folders inside sky_photos (e.g. haze/ and fog/).')
    paths, y, names = [], [], []
    for folder in label_folders:
        photo_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
        if len(photo_paths) < 5:
            raise ValueError(f'{folder.name}: Need at least five labeled images for a minimal train/test split.')
        names.append(folder.name)
        paths.extend(photo_paths)
        y.extend([folder.name]*len(photo_paths))
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(paths))
    train_i, test_i = train_test_split(idx, test_size=.25, stratify=y, random_state=seed)
    embeddings = image_embeddings(paths)
    labels = np.asarray(y)
    prototypes = []
    for name in names:
        prototype = embeddings[train_i][labels[train_i] == name].mean(axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-9)
        prototypes.append(prototype)
    prototypes = np.stack(prototypes)
    scores = embeddings[test_i] @ prototypes.T
    predicted = np.asarray(names)[scores.argmax(axis=1)]
    accuracy = float((predicted == labels[test_i]).mean())
    print(f'Holdout images: {len(test_i)} | Accuracy: {accuracy:.1%}')
    print('Small tests do not prove real-world performance. Evaluate on photos from different days/devices/locations.')
    np.savez_compressed(output, labels=np.asarray(names), prototypes=prototypes, holdout_accuracy=accuracy)
    print(f'Saved {output}. This does not estimate AQI or PM2.5.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sky_photos')
    parser.add_argument('--output', default='sky_prototypes.npz')
    args = parser.parse_args()
    train(args.data, args.output)
