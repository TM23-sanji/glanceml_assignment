import sys
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

MODEL_NAME = "patrickjohncyh/fashion-clip"


class FashionSearcher:
    def __init__(self, processed_dir="data/processed"):
        import faiss
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        self.index = faiss.read_index(f"{processed_dir}/faiss_index.bin")
        self.image_paths = np.load(
            f"{processed_dir}/image_paths.npy", allow_pickle=True
        )
        self.embeddings = np.load(f"{processed_dir}/image_embeddings.npy")

        self.attr_clothing = np.load(f"{processed_dir}/attr_clothing.npy")
        self.attr_colors = np.load(f"{processed_dir}/attr_colors.npy")
        self.attr_environment = np.load(f"{processed_dir}/attr_environment.npy")
        self.attr_style = np.load(f"{processed_dir}/attr_style.npy")

        self.clothing_labels = np.load(
            f"{processed_dir}/attr_clothing_labels.npy", allow_pickle=True
        ).tolist()
        self.colors_labels = np.load(
            f"{processed_dir}/attr_colors_labels.npy", allow_pickle=True
        ).tolist()
        self.environment_labels = np.load(
            f"{processed_dir}/attr_environment_labels.npy", allow_pickle=True
        ).tolist()
        self.style_labels = np.load(
            f"{processed_dir}/attr_style_labels.npy", allow_pickle=True
        ).tolist()

    @torch.inference_mode()
    def _encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            self.device
        )
        emb = self.model.get_text_features(**inputs)
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()

    def search(self, query: str, parsed_query, top_n: int = 200):
        query_emb = self._encode_text([query])

        scores, indices = self.index.search(query_emb.astype(np.float32), top_n)
        scores = scores[0]
        indices = indices[0]

        clip_sims = scores.copy()
        atomic_sims = self._compute_atomic_similarities(parsed_query, indices)
        attr_scores = self._compute_attr_consistency(parsed_query, indices)

        clip_sims = (clip_sims - clip_sims.min()) / (
            clip_sims.max() - clip_sims.min() + 1e-8
        )
        atomic_sims = (atomic_sims - atomic_sims.min()) / (
            atomic_sims.max() - atomic_sims.min() + 1e-8
        )
        attr_scores = (attr_scores - attr_scores.min()) / (
            attr_scores.max() - attr_scores.min() + 1e-8
        )

        final_scores = 0.4 * clip_sims + 0.35 * atomic_sims + 0.25 * attr_scores

        ranked = sorted(
            zip(indices, final_scores, clip_sims, atomic_sims, attr_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "ranked": ranked,
            "clip_similarities": clip_sims,
            "atomic_similarities": atomic_sims,
            "attr_consistencies": attr_scores,
        }

    def _compute_atomic_similarities(self, parsed_query, indices):
        atomic_texts = []
        for c in parsed_query.clothing:
            if c.color:
                atomic_texts.append(f"{c.color} {c.item}")
            else:
                atomic_texts.append(c.item)
        if parsed_query.environment:
            atomic_texts.append(f"photo in {parsed_query.environment}")
        if parsed_query.style:
            atomic_texts.append(f"{parsed_query.style} clothing")
        if parsed_query.activity:
            atomic_texts.append(parsed_query.activity)

        if not atomic_texts:
            return np.ones(len(indices))

        text_embs = self._encode_text(atomic_texts)
        candidate_embs = self.embeddings[indices]
        sims = candidate_embs @ text_embs.T
        return sims.mean(axis=1)

    def _compute_attr_consistency(self, parsed_query, indices):
        scores = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            consistency_score = 0.0
            count = 0

            for c in parsed_query.clothing:
                if c.color and c.color not in ("null", "none"):
                    item_idx = self._get_label_index(self.clothing_labels, c.item)
                    color_idx = self._get_label_index(self.colors_labels, c.color)
                    if item_idx >= 0 and color_idx >= 0:
                        item_score = float(self.attr_clothing[idx][item_idx])
                        color_score = float(self.attr_colors[idx][color_idx])
                        consistency_score += (item_score + color_score) / 2
                        count += 1

            if parsed_query.environment:
                env_idx = self._get_label_index(
                    self.environment_labels, parsed_query.environment
                )
                if env_idx >= 0:
                    consistency_score += float(self.attr_environment[idx][env_idx])
                    count += 1

            if parsed_query.style:
                style_idx = self._get_label_index(
                    self.style_labels, parsed_query.style
                )
                if style_idx >= 0:
                    consistency_score += float(self.attr_style[idx][style_idx])
                    count += 1

            scores[i] = consistency_score / max(count, 1)

        return scores

    @staticmethod
    def _get_label_index(labels, target):
        target_lower = target.lower().replace("_", " ").replace("-", " ")
        for i, label in enumerate(labels):
            if label.lower() == target_lower:
                return i
        return -1
