# constants.py
# --------------------------------
# Define your local setup paths here
IMAGE_DIR = r"C:\Users\MAXIMUS8\Downloads\archive\fashion-dataset\images"
JSON_DIR = r"C:\Users\MAXIMUS8\Downloads\archive\fashion-dataset\styles" # Assuming this contains individual .json files

# Output files (will be saved in the script's current working directory)
DATASET_TSV_PATH = 'fashion_dataset.tsv'
EMBEDDINGS_NPZ_PATH = 'fashion_embeddings.npz'

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "fashion_products_siglip2_naflex"

# Model Configuration
MODEL_ID = "google/siglip2-so400m-patch16-naflex"
DEVICE = "cuda" # "cuda" or "cpu"