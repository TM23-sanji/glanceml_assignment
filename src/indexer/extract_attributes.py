import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/processed"
BATCH_SIZE = 32
MODEL_NAME = "patrickjohncyh/fashion-clip"

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


@torch.inference_mode()
def extract_attributes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

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
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_embs = model.get_text_features(**text_inputs).pooler_output
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        all_scores = []
        for i in range(0, len(image_paths_str), BATCH_SIZE):
            batch = image_paths_str[i : i + BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch]
            img_inputs = processor(images=images, return_tensors="pt").to(device)
            img_embs = model.get_image_features(**img_inputs).pooler_output
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

            scores = (img_embs @ text_embs.T).cpu().numpy()
            all_scores.append(scores)

            if (i // BATCH_SIZE) % 5 == 0:
                print(f"  Processed {min(i + BATCH_SIZE, len(image_paths_str))}/{len(image_paths_str)}")

        all_scores = np.vstack(all_scores)
        np.save(f"{OUTPUT_DIR}/attr_{category}.npy", all_scores)
        print(f"  attr_{category}.npy shape: {all_scores.shape}")

    for category, labels in label_sets.items():
        np.save(f"{OUTPUT_DIR}/attr_{category}_labels.npy", np.array(labels))
    print("Done extracting all attributes")


if __name__ == "__main__":
    extract_attributes()
