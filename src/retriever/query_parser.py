import sys
import json
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.schemas import ParsedQuery, ClothingItem


class QueryParser:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Query parser loading on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def parse(self, query: str) -> ParsedQuery:
        messages = [
            {
                "role": "system",
                "content": "You extract structured data from fashion queries. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": f"""Extract fashion attributes from this query.
Return ONLY valid JSON matching:
{{"clothing": [{{"item": "type", "color": "color or null"}}], "environment": "place or null", "style": "style or null", "activity": "activity or null"}}

Query: {query}
JSON:""",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        response = response.strip()

        json_str = re.sub(r"^```json\s*|\s*```$", "", response).strip()
        json_str = re.sub(r"^```\s*|\s*```$", "", json_str).strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(json_str[start:end])
            else:
                data = {"clothing": [], "environment": None, "style": None, "activity": None}

        items = []
        for c in data.get("clothing", []):
            color = c.get("color")
            if color and color.lower() in ("null", "none", ""):
                color = None
            items.append(ClothingItem(item=c["item"], color=color))

        return ParsedQuery(
            clothing=items,
            environment=data.get("environment"),
            style=data.get("style"),
            activity=data.get("activity"),
        )
