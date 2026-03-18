import os
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models # Use models for newer features

from constants import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDINGS_NPZ_PATH
)

# Cache the client
_qdrant_client: QdrantClient = None

def get_qdrant_client() -> QdrantClient:
    """Initializes and returns a Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )
        # Sanity check to ensure connection
        try:
            _qdrant_client.get_collections()
            print(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}.")
        except Exception as e:
            print(f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. Please ensure Qdrant is running.")
            raise e # Re-raise the exception

    return _qdrant_client

def recreate_qdrant_collection(image_vector_size: int, text_vector_size: int):
    """Recreates the Qdrant collection with named vectors."""
    client = get_qdrant_client()

    # Delete collection if it already exists
    if client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        print(f"Deleting existing collection: {QDRANT_COLLECTION_NAME}")
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

    # Define configurations for multiple named vectors
    vectors_config: Dict[str, models.VectorParams] = {
        "image_vector": models.VectorParams(size=image_vector_size, distance=models.Distance.COSINE),
        "text_vector": models.VectorParams(size=text_vector_size, distance=models.Distance.COSINE)
    }

    # Create the new collection with named vectors
    print(f"Creating collection: {QDRANT_COLLECTION_NAME} with vectors_config: {vectors_config}")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=vectors_config
    )
    print(f"Collection '{QDRANT_COLLECTION_NAME}' created successfully.")


def upsert_embeddings_to_qdrant(batch_size: int = 256):
    """Reads embeddings from NPZ and upserts to Qdrant with named vectors."""
    client = get_qdrant_client()

    if not os.path.exists(EMBEDDINGS_NPZ_PATH):
        raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_NPZ_PATH}")

    print(f"Loading embeddings from {EMBEDDINGS_NPZ_PATH}")
    try:
        data = np.load(EMBEDDINGS_NPZ_PATH, allow_pickle=True)
        ids  = data["product_ids"]
        paths = data["image_paths"]
        img_e = data["image_embeddings"]
        txt_e = data["text_embeddings"]
    except Exception as e:
        print(f"Error loading data from NPZ file: {e}")
        print("Please ensure embeddings.npz contains 'product_ids', 'image_paths', 'image_embeddings', and 'text_embeddings'.")
        raise e

    # Validate shapes and get dimensions
    N, image_vector_size = img_e.shape
    N2, text_vector_size = txt_e.shape
    assert N == N2, f"Count mismatch between image ({N}) & text ({N2}) embeddings"
    print(f"Loaded {N} embeddings with image dim {image_vector_size} and text dim {text_vector_size}.")


    # Recreate collection with separate vector sizes
    recreate_qdrant_collection(image_vector_size=image_vector_size, text_vector_size=text_vector_size)

    # Build and upsert points with separate named vectors
    points: List[models.PointStruct] = []
    print("Preparing points for upsertion...")
    # Assuming product_ids are unique and can be used as Qdrant internal IDs
    # If product_ids are not integers, you might need a mapping or use UUIDs
    # However, using original product_ids as payload is generally good practice
    for i in range(N):
        # Ensure vectors are lists of floats
        image_vector = img_e[i].tolist()
        text_vector = txt_e[i].tolist()

        points.append(models.PointStruct(
            id=int(ids[i]), # Use integer product ID as Qdrant point ID if suitable
            vector={ # Store vectors as a dictionary with names
                "image_vector": image_vector,
                "text_vector": text_vector
            },
            payload={
                "product_id": str(ids[i]), # Store original ID as string in payload for retrieval
                "image_path": str(paths[i])
                # Add other payload fields if available in your data ingestion
                # e.g., "name": data["names"][i], "description": data["descriptions"][i]
            }
        ))

    print(f"Upserting {N} points to Qdrant in batches of {batch_size}...")
    # Use client.upsert with batch
    for i in tqdm(range(0, N, batch_size), desc="Upserting"):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=batch,
            wait=True # Wait for the operation to complete
        )
    print("Upsertion complete.")


def search_qdrant(query_embedding: np.ndarray, query_vector_name: str, top_k: int = 5):
    """Performs a search against a specific named vector in Qdrant."""
    client = get_qdrant_client()
    print(f"Searching collection '{QDRANT_COLLECTION_NAME}' with query_vector_name='{query_vector_name}'...")
    try:
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=models.NamedVector( # Use NamedVector for searching
                name=query_vector_name,
                vector=query_embedding.tolist() # Ensure query_embedding is a list of floats
            ),
            limit=top_k,
            with_payload=True # Retrieve payload along with search results
            # with_vectors=False # Set to True if you need vectors in the results in results
        )
        print(f"Found {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error during Qdrant search: {e}")
        # Provide more specific error info if possible
        try:
            collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            if isinstance(collection_info.vectors_config, models.VectorsConfigMap):
                 if query_vector_name not in collection_info.vectors_config.params_map:
                      print(f"Error: Query vector name '{query_vector_name}' does not exist in collection '{QDRANT_COLLECTION_NAME}'.")
                      print(f"Available vector names: {list(collection_info.vectors_config.params_map.keys())}")
                 # Check if vector size matches
                 elif len(query_embedding) != collection_info.vectors_config.params_map[query_vector_name].size:
                     print(f"Error: Query vector size ({len(query_embedding)}) does not match stored vector size ({collection_info.vectors_config.params_map[query_vector_name].size}) for vector '{query_vector_name}'.")
            else:
                 print(f"Error: Collection '{QDRANT_COLLECTION_NAME}' is not configured with named vectors.")

        except Exception as info_e:
             print(f"Could not retrieve collection info for detailed error checking: {info_e}")

        raise e # Re-raise the exception

if __name__ == "__main__":
    # Example usage: Run the upsertion process when the script is executed directly
    # Make sure your embeddings.npz file is ready before running this.
    try:
        upsert_embeddings_to_qdrant(batch_size=256)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate the embeddings.npz file first by running your data ingestion script.")
    except Exception as e:
        print(f"An error occurred during upsertion: {e}")