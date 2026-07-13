import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/processed"
BATCH_SIZE = 32

CLOTHING_ITEMS = [
    "shirt", "t-shirt", "blouse", "sweater", "cardigan",
    "hoodie", "sweatshirt", "jacket", "coat", "blazer",
    "suit", "vest", "tie", "dress", "skirt",
    "pants", "jeans", "shorts", "leggings", "overalls",
    "jumpsuit", "tank top", "crop top", "polo", "button-down shirt",
]

COLORS = [
    "red", "blue", "green", "yellow", "black",
    "white", "gray", "brown", "pink", "purple",
    "orange", "beige", "navy", "teal", "maroon",
    "olive", "coral", "turquoise", "gold", "silver",
]

ENVIRONMENTS = [
    "indoor", "outdoor", "office", "park",
    "urban street", "home", "beach", "restaurant",
]

STYLES = [
    "formal", "casual", "sporty", "vintage",
    "minimalist", "elegant", "bohemian", "preppy",
]


def extract_attributes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from fashion_clip.fashion_clip import FashionCLIP
    fclip = FashionCLIP("patrickjohncyh/fashion-clip")

    image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
    image_paths_str = [str(p) for p in image_paths]

    prompt_sets = {
        "clothing": [f"a {item}" for item in CLOTHING_ITEMS],
        "colors": COLORS,
        "environment": [f"photo in {env}" for env in ENVIRONMENTS],
        "style": [f"{style} clothing" for style in STYLES],
    }

    label_sets = {
        "clothing": CLOTHING_ITEMS,
        "colors": COLORS,
        "environment": ENVIRONMENTS,
        "style": STYLES,
    }

    for category, prompts in prompt_sets.items():
        print(f"Extracting {category} attributes...")
        text_embs = fclip.encode_text(prompts, batch_size=BATCH_SIZE)
        text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

        all_scores = []
        for i in range(0, len(image_paths_str), BATCH_SIZE):
            batch = image_paths_str[i : i + BATCH_SIZE]
            img_embs = fclip.encode_images(batch, batch_size=BATCH_SIZE)
            img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)
            scores = img_embs @ text_embs.T
            all_scores.append(scores)

        all_scores = np.vstack(all_scores)
        np.save(f"{OUTPUT_DIR}/attr_{category}.npy", all_scores)
        print(f"  attr_{category}.npy shape: {all_scores.shape}")

    for category, labels in label_sets.items():
        np.save(f"{OUTPUT_DIR}/attr_{category}_labels.npy", np.array(labels))
    print("Done extracting all attributes")


if __name__ == "__main__":
    extract_attributes()
