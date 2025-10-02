# 🔎 Local NLP Paper Search (Asta-esque Tool)

This project provides a lightweight, local research discovery tool inspired by **AllenAI’s Asta**.  
It lets you generate embeddings for scientific papers with **SPECTER2**, build a searchable index, and interactively explore results via a **Streamlit app**.  

---

## 📂 Project Structure

### 1. `SPECTER2_embedding_gen.py`  
**Purpose**: Generates dense vector embeddings for research papers (title + abstract) using **SPECTER2**.  

**Workflow**:  
1. Loads metadata from `arxiv_metadata.csv`.  
2. Concatenates each paper’s title and abstract.  
3. Encodes them with SPECTER2 (via Hugging Face + adapters).  
4. Outputs a **Parquet file** (`arxiv_specter2_embeddings.parquet`) containing embeddings + metadata.  

👉 This script builds your **corpus embeddings**, the foundation for search and retrieval.  

---

### 2. `app_with_metadata.py`  
**Purpose**: Provides a **Streamlit-based search interface** for exploring the embedded paper collection.  

**Features**:  
- Loads embeddings from `arxiv_specter2_embeddings.parquet` and optional metadata from `arxiv_metadata.csv`.  
- Uses **FAISS** for fast Approximate Nearest Neighbor (ANN) search.  
- Optionally re-ranks results with **mxbai embedding reranker** for better relevance.  
- Integrates an **open-source LLM (Qwen)** to explain *why a paper is relevant* to your query.  
- Displays metadata (DOI, arXiv ID, published/updated dates, filenames, etc.) alongside results.  

👉 This app is your **interactive search front-end**, mimicking how Asta surfaces and contextualizes relevant papers.  

---

## 🚀 Usage

### 1. Generate embeddings
```bash
python SPECTER2_embedding_gen.py
