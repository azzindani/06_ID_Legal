# Loader Module

Dataset loading and preprocessing for the Indonesian Legal RAG System.

## Components

| File | Description |
|------|-------------|
| `dataloader.py` | Enhanced dataset loader with KG support |

## Usage

```python
from loader import EnhancedKGDatasetLoader

# Initialize loader
loader = EnhancedKGDatasetLoader(
    dataset_name='Azzindani/ID_REG_DB_2510',
    embedding_model=model
)

# Load and process dataset
loader.load_dataset()
loader.create_embeddings()
loader.build_faiss_index()

# Search
results = loader.search(query_embedding, top_k=10)
```

## Dataset Structure

Expected columns:
- `content` - Document text
- `regulation_type` - UU, PP, etc.
- `regulation_number` - Number
- `year` - Year
- `about` - Subject matter

## Features

- Automatic embedding generation
- FAISS index building
- Keyword index (BM25)
- Hybrid search support
