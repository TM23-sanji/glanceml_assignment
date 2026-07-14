import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image
from src.retriever.pipeline import RetrievalPipeline


@st.cache_resource
def load_pipeline():
    return RetrievalPipeline()


st.set_page_config(page_title="Fashion Retrieval", layout="wide")
st.title("Fashion & Context Retrieval")

pipeline = load_pipeline()

query = st.text_input(
    "Search for fashion:",
    placeholder="e.g., A person in a bright yellow raincoat",
)

col1, _ = st.columns([1, 5])
with col1:
    k = st.number_input("Results", min_value=1, max_value=50, value=10)

if query:
    with st.spinner("Searching..."):
        parsed, results = pipeline.query(query, k=k)

    with st.expander("Parsed query", expanded=False):
        st.json(parsed.model_dump())

    if results:
        st.success(f"Top {len(results)} results")

        cols = st.columns(min(5, len(results)))
        for i, result in enumerate(results):
            with cols[i % 5]:
                img = Image.open(result.path)
                st.image(img, width=300)
                st.caption(
                    f"Score: {result.score:.3f} | CLIP: {result.clip_similarity:.2f} | "
                    f"Atomic: {result.atomic_similarity:.2f} | Attr: {result.attr_consistency:.2f}"
                )
    else:
        st.warning("No results found")
