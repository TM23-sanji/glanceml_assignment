# Multimodal Fashion & Context Retrieval

Compositional fashion image retrieval using FashionCLIP + query decomposition + attribute consistency re-ranking.

## Architecture

- **Indexer**: FashionCLIP embeddings + zero-shot attribute extraction → FAISS
- **Retriever**: Qwen2.5-1.5B query parsing → FAISS coarse search → compositional re-ranking

## Setup

```bash
uv sync
```

## Usage

### 1. Index images

```bash
python -m src.indexer.extract_embeddings
python -m src.indexer.extract_attributes
python -m src.indexer.build_index
```

### 2. Launch UI

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
src/
  indexer/          # Feature extraction & indexing
    extract_embeddings.py
    extract_attributes.py
    build_index.py
  retriever/        # Query parsing & search
    query_parser.py     # Qwen2.5-1.5B + Pydantic
    search.py           # FAISS + re-ranker
    pipeline.py         # End-to-end
  models/
    schemas.py          # Pydantic models
app/
  streamlit_app.py
```
