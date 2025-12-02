# GraphNN Yambda Recommendation System

This repository contains the implementation of Graph Neural Network (GNN) based recommendation systems for the **Yambda-50m** music dataset. The project focuses on comparing Multi-Behavior Graph Convolutional Networks (**MBGCN**) against state-of-the-art baselines like **LightGCN** using a modular and scalable approach.

The codebase is built with **PyTorch Geometric**, managed by **Hydra** for configuration, and integrated with **ClearML** for experiment tracking.

## 🚀 Key Features

- **Multi-Behavior Modeling (MBGCN):** Exploits various interaction types (`listen+`, `like`, `dislike`, etc.) to improve recommendation quality for a specific target event.
    
- **Scalable Training:** Utilizes `LinkNeighborLoader` from PyTorch Geometric to efficiently train on large-scale graphs (50M+ edges) using subgraph sampling.
    
- **Modular Architecture:** Easy switching between models (`MBGCN`, `LightGCN`) and training strategies without code changes.
    
- **Content-Aware:** Supports initialization with pre-trained semantic track embeddings (e.g., from content-based models).
    
- **Experiment Management:** Full control over hyperparameters via Hydra CLI and automatic logging to ClearML.
    

## 📂 Repository Structure

The core logic resides in the `src/` directory.

```
src/
├── configs/                 # Hydra configuration files (.yaml)
│   ├── model/               # Model-specific configs (mbgcn.yaml, lightgcn.yaml)
│   ├── dataset/             # Data paths and processing rules
│   ├── dataloader/          # Batch size and neighbor sampling settings
│   ├── trainer/             # Training loop settings (epochs, val_interval)
│   └── train.yaml           # Main configuration entry point
├── datasets/
│   ├── dataloader.py        # PyG Data construction and LinkNeighborLoader setup
│   └── preprocessor.py      # Parquet loading, 'listen' -> 'listen+' logic
├── model/
│   ├── mbgcn.py             # Multi-Behavior GCN implementation
│   └── lightgcn.py          # LightGCN implementation
├── trainer/
│   ├── trainer.py           # MBGCN training loop
│   └── lightgcn_trainer.py  # LightGCN training loop (homogeneous graph filtering)
├── metrics/                 # Ranking metrics (NDCG, Recall)
├── loss/                    # BPR Loss implementation
├── logger/                  # ClearML experiment logger wrapper
└── train.py                 # Main entry point for training and evaluation

```

## 🛠 Installation

1. **Clone the repository:**
    
    ```
    git clone <repository_url>
    cd GraphNN-Yambda-RecommendationSystem
    ```
    
2. **Install dependencies:** Ensure you have Python 3.8+ installed.
    
    ```
    pip install -r requirements.txt
    ```
    
    > Note: Ensure your `torch` and `torch-geometric` versions match your CUDA version.
    

## 💾 Data Preparation

The pre-processed data for this project is available on Kaggle. You do not need to perform raw data processing manually.

1. **Download the Dataset:** Get the prepared data from the following link: 👉 [**Yambda-50m Compressed Dataset**](https://www.kaggle.com/datasets/dinaryakupov/yambda-50m-compressed "null")
    
2. **Directory Structure:** Place the downloaded `yambda` folder inside the `data/` directory. The structure should look exactly like this:
    
    ```
    src/
    ...
    data/
    └── yambda/
        ├── embeddings.parquet           # Pre-trained item embeddings (Optional but recommended)
        └── flat/
            └── 50m/
                ├── multi_event_train.parquet
                ├── multi_event_val.parquet
                └── multi_event_test.parquet
    ```
## 🧪 Running Experiments

All experiments are launched via `src/train.py`. You can configure **any** parameter from the command line using Hydra syntax.

### 1. Selecting the Model

Use the `model` argument to switch architectures.

- **Train MBGCN (Default):**
    
    ```
    python src/train.py model=mbgcn
    
    ```
    
- **Train LightGCN:**
    
    ```
    python src/train.py model=lightgcn
    
    ```
    

### 2. Common Usage Scenarios

#### Scenario A: Quick Test (CPU)

Run a fast check to ensure the pipeline works, using a small data sample.

```
python src/train.py \
    device='cpu' \
    dataset.train_sampling_ratio=0.01 \
    dataloader.batch_size=128 \
    trainer.epochs=1

```

#### Scenario B: Full Training for 'listen+' Prediction

Train MBGCN to predict `listen+` events using all available data.

```
python src/train.py \
    model=mbgcn \
    target_event='listen+' \
    dataset.train_sampling_ratio=1.0 \
    trainer.epochs=20

```

#### Scenario C: LightGCN Baseline on 'like' Events

Train LightGCN specifically on `like` interactions. The trainer will automatically filter the graph to include only `like` edges for propagation.

```
python src/train.py \
    model=lightgcn \
    target_event='like' \
    model.embedding_dim=64 \
    model.n_layers=3 \
    trainer.epochs=50

```

## 📱 Demo Application

The repository includes a production-ready demo application designed to showcase the recommendation system in a real-world scenario. It is located in the `Application/` directory.

### Architecture

The application is built using a microservices architecture and containerized with Docker for easy deployment:

1. **Backend API (FastAPI):**
    
    - Located in `Application/code/deployment/api`.
        
    - Loads the trained model artifacts and serves recommendation predictions via REST endpoints.
        
    - Handles data processing and model inference.
        
2. **Frontend UI (Streamlit):**
    
    - Located in `Application/code/deployment/streamlit_app`.
        
    - Provides an interactive web interface for users to select users/items and view recommendations.
        
    - Communicates with the Backend API to fetch results.
        
3. **Orchestration:**
    
    - Managed by `docker-compose` to run both services simultaneously in a unified network.
        

### How to Run the App

1. Navigate to the application code directory:
    
    ```
    cd Application/code
    ```
    
2. Start the services using Docker Compose:
    
    ```
    docker-compose up --build
    ```
    
3. Access the services:
    
    - **Frontend Interface:** Open [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501 "null") in your browser.
        
    - **Backend API Docs:** Open [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs "null") to explore the Swagger UI.

## ⚙️ Configuration Guide

Parameters are defined in `src/configs/`. You can override them using dot notation (e.g., `group.option=value`).

### Global Parameters (`src/configs/train.yaml`)

|**Parameter**|**Description**|**Default**|
|---|---|---|
|`target_event`|The target interaction to predict (e.g., `listen+`, `like`).|`listen+`|
|`seed`|Random seed for reproducibility.|`42`|
|`device`|Compute device (`cuda` or `cpu`).|`cuda`|
|`experiment_name`|Name of the experiment for logging.|`mbgcn-yambda-{dataset.target_event}`|

### Dataset & Dataloader

|**Parameter**|**Description**|**Default**|
|---|---|---|
|`dataset.train_sampling_ratio`|Fraction of training data to use (0.0 - 1.0).|`1.0`|
|`dataset.embedding_path`|Path to parquet file with embeddings. Set to `null` to disable.|`.../embeddings.parquet`|
|`dataloader.batch_size`|Training batch size.|`1024`|
|`dataloader.num_neighbors`|List of neighbors to sample per layer (e.g., `[20, 10]`).|`[10, 5]`|

### Model Specifics

#### MBGCN (`model=mbgcn`)

|**Parameter**|**Description**|
|---|---|
|`model.n_gnn_layers`|Number of GNN layers per behavior block.|
|`model.fusion`|Fusion strategy: `attention`, `concat`, `sum`.|
|`model.behaviour_hidden_dim`|Hidden dimension size for behavior blocks.|

#### LightGCN (`model=lightgcn`)

|**Parameter**|**Description**|
|---|---|
|`model.n_layers`|Number of propagation layers.|
|`model.embedding_dim`|Size of user/item embeddings.|
|`model.item_feat_dim`|Input dimension of pretrained features (auto-detected).|

## 📊 Monitoring (ClearML)

The project is integrated with **ClearML** for experiment tracking.

1. **Metrics:** Real-time plots for Loss, NDCG@k, and Recall@k.
    
2. **Artifacts:** The best model checkpoint (`best_model.pt`) is automatically saved when the primary metric improves.
    
3. **Configuration:** The exact Hydra configuration used for the run is saved, ensuring reproducibility.
    

To disable or configure logging, edit `src/configs/logger/clearml.yaml`.

## 🧠 Implementation Details

- **Negative Sampling:**
    
    - Training: Implicitly handled via BPR Loss (1 positive vs. 1 negative). The ratio is fixed in the code to ensure correctness.
        
    - Evaluation: Full-ranking evaluation is performed (ranking the ground truth item against all other items).
        
- **Graph Construction:**
    
    - **MBGCN:** Uses a heterogeneous graph where edge types correspond to interaction types (`listen+`, `like`, etc.).
        
    - **LightGCN:** Effectively sees a homogeneous graph. The `LightGCNTrainer` filters the batch to keep only edges corresponding to the `target_event`.
        
- **Item Embeddings:**
    
    - If `dataset.embedding_path` is provided, models project these features into the embedding space.
        
    - If `dataset.embedding_path=null`, models initialize learnable embedding tables (ID-based).