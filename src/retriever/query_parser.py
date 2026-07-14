import sys
import json
import re
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from openai import OpenAI
from src.models.schemas import ParsedQuery, ClothingItem


class QueryParser:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )
        self.model = "meta-llama/Llama-3.1-8B-Instruct:deepinfra"

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

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=128,
            temperature=0,
        )
        json_str = completion.choices[0].message.content.strip()

        json_str = re.sub(r"^```json\s*|\s*```$", "", json_str).strip()
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
            item_name = c.get("item")
            if not item_name:
                continue
            color = c.get("color")
            if color and color.lower() in ("null", "none", ""):
                color = None
            items.append(ClothingItem(item=item_name, color=color))

        return ParsedQuery(
            clothing=items,
            environment=data.get("environment"),
            style=data.get("style"),
            activity=data.get("activity"),
        )
