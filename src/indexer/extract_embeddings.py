import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.schemas import ImageMeta

IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/processed"
BATCH_SIZE = 32


def extract_embeddings():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from fashion_clip.fashion_clip import FashionCLIP
    fclip = FashionCLIP("patrickjohncyh/fashion-clip")

    image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
    print(f"Found {len(image_paths)} images")

    image_paths_str = [str(p) for p in image_paths]
    embeddings = fclip.encode_images(image_paths_str, batch_size=BATCH_SIZE)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save(f"{OUTPUT_DIR}/image_embeddings.npy", embeddings)
    np.save(f"{OUTPUT_DIR}/image_paths.npy", image_paths_str)

    metas = [ImageMeta(path=p, faiss_id=i) for i, p in enumerate(image_paths_str)]
    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_embeddings()
