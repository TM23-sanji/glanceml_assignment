# Multimodal Fashion & Context Retrieval

Compositional fashion image retrieval using FashionCLIP + query decomposition + attribute consistency re-ranking.

## Architecture

- **Indexer**: FashionCLIP embeddings + zero-shot attribute extraction → FAISS
- **Retriever**: LLM (Llama 3.1 via HF API) query parsing → FAISS coarse search → compositional re-ranking

![Architecture Diagram](image.png)

## Setup

```bash
# Clone the repo
git clone https://github.com/TM23-sanji/glanceml_assignment.git
cd glanceml_assignment

# Download and extract fashion images
mkdir -p data/images
wget https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
unzip -o val_test2020.zip -d data/images/
rm val_test2020.zip
# The zip extracts into a 'test/' subfolder — move files up
mv data/images/test/* data/images/ && rmdir data/images/test

# Install dependencies
uv sync
```

## Usage

```bash
# Activate environment and set HF token (loads model weights locally)
export HF_TOKEN="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"

source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Each image is a `.jpg` file (3200 total, ~235 MB). The folder structure should be:

```
data/
  images/             # 3200 .jpg files (not tracked in git)
  processed/          # Pre-built FAISS index and embeddings (tracked in git)
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
