import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.schemas import ImageMeta

IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/processed"
BATCH_SIZE = 32
MODEL_NAME = "patrickjohncyh/fashion-clip"


@torch.inference_mode()
def extract_embeddings():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
    print(f"Found {len(image_paths)} images")

    all_embeddings = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)

        outputs = model.get_image_features(**inputs)
        emb = outputs.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)}")

    embeddings = np.vstack(all_embeddings)
    image_paths_str = [str(p) for p in image_paths]

    np.save(f"{OUTPUT_DIR}/image_embeddings.npy", embeddings)
    np.save(f"{OUTPUT_DIR}/image_paths.npy", image_paths_str)

    metas = [ImageMeta(path=p, faiss_id=i) for i, p in enumerate(image_paths_str)]
    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_embeddings()
