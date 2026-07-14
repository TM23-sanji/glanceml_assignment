import os
import sys
import numpy as np
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

OUTPUT_DIR = "data/processed"


def build_index():
    embeddings_path = f"{OUTPUT_DIR}/image_embeddings.npy"
    paths_path = f"{OUTPUT_DIR}/image_paths.npy"

    if not os.path.exists(embeddings_path):
        print("Embeddings not found. Run extract_embeddings.py first.")
        return

    embeddings = np.load(embeddings_path)
    image_paths = np.load(paths_path, allow_pickle=True)

    import faiss

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, f"{OUTPUT_DIR}/faiss_index.bin")

    metadata = {
        "num_images": int(len(image_paths)),
        "dimension": int(dimension),
        "index_type": "IndexFlatIP",
        "image_paths": [str(p) for p in image_paths],
    }
    with open(f"{OUTPUT_DIR}/index_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Index built: {len(image_paths)} images, {dimension}-d vectors")
    print(f"Saved to {OUTPUT_DIR}/faiss_index.bin")


if __name__ == "__main__":
    build_index()
