# app.py

import os
import streamlit as st
from PIL import Image
import numpy as np
# Add this line immediately after importing torch to fix the known Streamlit/torch.classes issue
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    st.error("PyTorch is not installed. Please install it to run this app.")
    # Consider adding a graceful exit or disabling PyTorch-dependent features
    torch = None # Set torch to None if import fails to prevent further errors

import time # Import time for potential timing or delays

# Only import other modules if torch was successfully imported, or handle appropriately
if torch is not None:
    import constants # Import constants to access QDRANT_COLLECTION_NAME
    from constants import DEVICE, IMAGE_DIR, EMBEDDINGS_NPZ_PATH
    from embed_utils import get_siglip_models_and_processor
    # Import the modified qdrant_ops functions
    from qdrant_ops import get_qdrant_client, search_qdrant, upsert_embeddings_to_qdrant
    from qdrant_client import models # Import models for checking collection config
    # Import necessary Qdrant models if they are used directly in app.py checks
    # from qdrant_client.http.models import CollectionInfo # No longer directly used this way in the fix


st.set_page_config(layout="wide")


# Function to perform upsert if embeddings.npz is missing or needs regeneration
# Updated to handle potential variations in CollectionInfo structure
def ensure_embeddings_in_qdrant():
    """
    Checks if embeddings are in Qdrant with the correct configuration (named vectors).
    Attempts to upsert if missing or incorrect.
    Returns True if Qdrant is ready, False otherwise.
    """
    if not os.path.exists(EMBEDDINGS_NPZ_PATH):
        st.error(f"Embeddings file not found at {EMBEDDINGS_NPZ_PATH}. Please run the data ingestion script first.")
        return False

    try:
        client = get_qdrant_client()
        collection_name = constants.QDRANT_COLLECTION_NAME

        # Check if the collection exists
        if not client.collection_exists(collection_name=collection_name):
             print(f"Qdrant collection '{collection_name}' not found. Attempting to upsert embeddings.") # Use print, st not available in cached fn early runs
             upsert_embeddings_to_qdrant()
             print("Embeddings upserted successfully.")
             # Add a small delay to allow Qdrant to index
             time.sleep(2) # Adjust delay as needed
             return True # Qdrant is now ready

        # Get collection info to check its configuration
        collection_info = client.get_collection(collection_name=collection_name)

        # --- FIX for 'CollectionInfo' object has no attribute 'vectors_config' ---
        # Check if 'vectors_config' attribute exists and is a VectorsConfigMap (for named vectors)
        if not (hasattr(collection_info, 'vectors_config') and isinstance(collection_info.vectors_config, models.VectorsConfigMap)):
            print(f"Qdrant collection '{collection_name}' does not appear to have a named vector configuration. Attempting to re-upsert embeddings.")
            upsert_embeddings_to_qdrant()
            print("Embeddings upserted successfully.")
            time.sleep(2) # Adjust delay as needed
            return True # Qdrant is now ready

        # Now that we've confirmed it's likely a VectorsConfigMap, check for the expected named vectors
        # Access the parameters map from the vectors_config
        vector_params_map = collection_info.vectors_config.params_map

        # Check if the expected named vectors exist and have the correct sizes
        # We need to load the expected sizes from the embeddings file here
        if not os.path.exists(EMBEDDINGS_NPZ_PATH):
             print(f"Error: Embeddings file not found at {EMBEDDINGS_NPZ_PATH} during size check.")
             return False

        try:
             data = np.load(EMBEDDINGS_NPZ_PATH, allow_pickle=True)
             expected_img_dim = data["image_embeddings"].shape[1]
             expected_txt_dim = data["text_embeddings"].shape[1]
        except Exception as e:
             print(f"Error loading embedding dimensions from NPZ: {e}")
             return False


        expected_vector_configs = {
            "image_vector": expected_img_dim,
            "text_vector": expected_txt_dim
        }

        config_mismatch = False
        if set(vector_params_map.keys()) != set(expected_vector_configs.keys()):
             config_mismatch = True
        else:
            for name, size in expected_vector_configs.items():
                 if vector_params_map[name].size != size:
                     config_mismatch = True
                     break

        if config_mismatch:
             print(f"Qdrant collection '{collection_name}' vector configuration mismatch. Expected: {expected_vector_configs}, Found: {{name: params.size for name, params in vector_params_map.items()}}. Attempting to re-upsert embeddings.")
             upsert_embeddings_to_qdrant()
             print("Embeddings upserted successfully.")
             time.sleep(2) # Adjust delay as needed
             return True # Qdrant is now ready


        # Optionally, check if the number of points matches the embeddings file
        # This can be slow for large collections, consider commenting out if performance is an issue
        # try:
        #     data = np.load(EMBEDDINGS_NPZ_PATH, allow_pickle=True)
        #     expected_points = len(data["product_ids"])
        #     if collection_info.points_count < expected_points:
        #          print(f"Qdrant collection '{collection_name}' has fewer points ({collection_info.points_count}) than embeddings file ({expected_points}). Attempting to re-upsert embeddings.")
        #          upsert_embeddings_to_qdrant()
        #          print("Embeddings upserted successfully.")
        #          time.sleep(2) # Adjust delay
        #          return True # Qdrant is now ready
        # except Exception as e:
        #      print(f"Error checking point count: {e}")
        #      # Decide if you want to fail or continue if point count check fails
        #      pass


        print(f"Qdrant collection '{collection_name}' is configured correctly and appears populated.")
        return True # Collection exists and looks correct

    except FileNotFoundError:
         # This case is handled at the beginning of the function
         return False
    except Exception as e:
        print(f"Error checking or upserting embeddings to Qdrant: {e}")
        # Provide guidance to check Qdrant status
        print("Please ensure your Qdrant instance is running and accessible.")
        # Use st.error here if possible, but this function is cached, so errors might not show immediately
        # st.error(f"Error checking or upserting embeddings to Qdrant: {e}")
        return False


# --- Initial setup wrapped in a cached function ---
@st.cache_resource
def initialize_app_resources():
    """
    Initializes all necessary resources: checks Qdrant, loads models and dimensions.
    This function is cached by Streamlit.
    """
    print("Attempting to initialize app resources...")
    if ensure_embeddings_in_qdrant():
        try:
            processor, vision_model, text_model = get_siglip_models_and_processor(device=DEVICE)
            print("SigLIP-2 models loaded within cached function.")

            if not os.path.exists(EMBEDDINGS_NPZ_PATH):
                 print(f"Error: Embeddings NPZ file not found at {EMBEDDINGS_NPZ_PATH} during resource loading.")
                 return None, None, None, None, None, None

            data = np.load(EMBEDDINGS_NPZ_PATH, allow_pickle=True)
            IMG_DIM = data["image_embeddings"].shape[1]
            TXT_DIM = data["text_embeddings"].shape[1]
            print(f"Read embedding dimensions within cached function: Image={IMG_DIM}, Text={TXT_DIM}.")

            qdrant_client = get_qdrant_client()
            print("Qdrant client connected within cached function.")

            print("App resources initialized successfully.")
            return processor, vision_model, text_model, IMG_DIM, TXT_DIM, qdrant_client

        except Exception as e:
            print(f"Error loading resources within cached function: {e}")
            # st.error is not available directly in cached functions early on
            return None, None, None, None, None, None # Indicate failure
    else:
        print("Qdrant not ready. Initialization failed.")
        return None, None, None, None, None, None # Indicate failure

# Call the cached initialization function
processor, vision_model, text_model, IMG_DIM, TXT_DIM, qdrant_client = initialize_app_resources()

# Check if initialization was successful outside the cached function
models_loaded = processor is not None

# --- Rest of your app logic ---
# Only proceed if torch was imported successfully and models are loaded
if torch is not None and models_loaded:

    # --- Embedding functions (adapted for SigLIP2 output) ---
    # Moved inside the loaded check as they depend on processor/models
    def embed_query_image(image: Image.Image, processor, vision_model) -> np.ndarray:
        """Generates a normalized embedding for a query image using the vision model."""
        inputs = processor(
            images=[image],
            return_tensors="pt",
            padding=True # Ensure padding is True to potentially get attention mask
        ).to(DEVICE)

        with torch.no_grad():
            # Pass the required arguments from the inputs dictionary to the model
            # Check if the inputs object has these attributes before accessing them
            model_inputs = {
                "pixel_values": inputs.pixel_values,
            }
            if hasattr(inputs, 'pixel_attention_mask'):
                 model_inputs["pixel_attention_mask"] = inputs.pixel_attention_mask
            if hasattr(inputs, 'spatial_shapes'):
                 model_inputs["spatial_shapes"] = inputs.spatial_shapes
            # Note: Depending on the exact model/processor version, spatial_shapes might not
            # be directly in inputs. If you get a new error about spatial_shapes
            # not being in inputs, you might need to check the processor's documentation
            # or construct it from pixel_values.shape. But let's try passing it if it exists.


            out = vision_model(**model_inputs) # Use ** to unpack the dictionary

        emb = None
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
             emb = out.pooler_output.cpu().squeeze().numpy()
        elif hasattr(out, 'last_hidden_state'):
             emb = out.last_hidden_state[:, 0, :].cpu().squeeze().numpy()
        else:
             raise AttributeError("Could not find a suitable pooled output in the vision model output.")

        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def embed_query_text(text: str, processor, text_model) -> np.ndarray:
        """Generates a normalized embedding for a query text using the text model."""
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64, # Ensure this matches your ingestion if you set a max_length there
            return_attention_mask=True
        ).to(DEVICE)

        input_ids       = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            out = text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        emb = None
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
             emb = out.pooler_output.cpu().squeeze().numpy()
        elif hasattr(out, 'last_hidden_state'):
             emb = out.last_hidden_state[:, 0, :].cpu().squeeze().numpy()
        else:
             raise AttributeError("Could not find a suitable pooled output in the text model output.")


        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


    st.title("✨ SigLIP2 Fashion Search (Local Qdrant)")

    # Rest of your Streamlit app UI and search logic
    mode = st.sidebar.radio("Search by:", ["Image", "Text", "Both"])
    uploaded_image = None
    query_text     = ""

    if mode in ("Image", "Both"):
        img_file = st.sidebar.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])
        if img_file:
            uploaded_image = Image.open(img_file).convert("RGB")
            st.sidebar.image(uploaded_image, width=150)

    if mode in ("Text", "Both"):
        query_text = st.sidebar.text_input("Enter description:")

    top_k = st.sidebar.slider("Number of results:", 1, 12, 6)

    # Corrected indentation for the search button logic
    if st.sidebar.button("Search"):
        img_emb = None
        txt_emb = None
        hits = [] # Initialize hits list

        # Embed both modalities (if provided)
        if uploaded_image:
             try:
                 # Pass processor and models to embedding functions
                 img_emb = embed_query_image(uploaded_image, processor, vision_model)
             except Exception as e:
                 st.error(f"Error embedding image: {e}")
                 st.stop() # Stop execution if embedding fails

        if query_text:
             try:
                 # Pass processor and models to embedding functions
                 txt_emb = embed_query_text(query_text, processor, text_model)
             except Exception as e:
                 st.error(f"Error embedding text: {e}")
                 st.stop() # Stop execution if embedding fails

        # Validation
        if mode == "Image" and img_emb is None:
            st.error("Please upload an image for Image search."); st.stop()
        if mode == "Text"  and txt_emb is None:
            st.error("Please enter text for Text search.");     st.stop()
        if mode == "Both" and (img_emb is None or txt_emb is None):
             st.error("Please upload an image and enter text for 'Both' mode."); st.stop()

        # --- Perform search using named vectors based on mode ---
        with st.spinner(f"Searching Qdrant by {mode}..."):
            try:
                if mode == "Image" and img_emb is not None:
                    hits = search_qdrant(img_emb, query_vector_name="image_vector", top_k=top_k)
                elif mode == "Text" and txt_emb is not None:
                    hits = search_qdrant(txt_emb, query_vector_name="text_vector", top_k=top_k)
                elif mode == "Both" and img_emb is not None and txt_emb is not None:
                    # Strategy for "Both" mode: Search both vectors and combine results
                    # Get more hits than top_k from each search to improve merging quality
                    image_hits = search_qdrant(img_emb, query_vector_name="image_vector", top_k=top_k * 2)
                    text_hits = search_qdrant(txt_emb, query_vector_name="text_vector", top_k=top_k * 2)

                    # Simple merging strategy: Use a dictionary to track the best score for each product ID
                    # and the corresponding hit object.
                    merged_scores = {}
                    merged_hits_dict = {} # Use a dictionary to easily update/retrieve hits by product_id

                    for hit in image_hits:
                         pid = hit.payload.get("product_id")
                         score = hit.score
                         if pid not in merged_scores or score > merged_scores[pid]:
                              merged_scores[pid] = score
                              merged_hits_dict[pid] = hit # Store the hit with the current best score

                    for hit in text_hits:
                         pid = hit.payload.get("product_id")
                         score = hit.score
                         # If the product ID is already present, combine scores (e.g., take the max)
                         if pid in merged_scores:
                              merged_scores[pid] = max(merged_scores[pid], score)
                              # Optionally update the stored hit if the text score is higher, or
                              # implement a more complex merging of the hit objects if needed.
                              # For simplicity here, we just ensure the score is the max.
                              # If you need to track which modality gave the best score, store that too.
                         else:
                              merged_scores[pid] = score
                              merged_hits_dict[pid] = hit # Store the hit from text search

                    # Convert the dictionary values back to a list of hits
                    merged_hits_list = list(merged_hits_dict.values())

                    # Sort the unique hits by their merged score in descending order
                    sorted_merged_hits = sorted(merged_hits_list, key=lambda hit: merged_scores[hit.payload.get("product_id")], reverse=True)

                    # Take the top_k results from the merged and sorted list
                    hits = sorted_merged_hits[:top_k]

            except Exception as e:
                st.error(f"An error occurred during search: {e}")

        # This block was previously mis-indented inside the with st.spinner
        st.subheader("Search Results")
        if not hits:
            st.info("No matches found.")
        else:
            # Display results in columns
            cols = st.columns(min(len(hits), 4))
            for i, hit in enumerate(hits):
                # Use modulo operator to wrap around columns
                with cols[i % 4]:
                    pid  = hit.payload.get("product_id", "N/A")
                    path = hit.payload.get("image_path", "")
                    # Display the merged score for "Both" mode, or the individual score
                    # from the respective search for "Image" or "Text" mode.
                    display_score = merged_scores.get(pid, hit.score) if mode == "Both" else hit.score
                    st.markdown(f"**ID:** {pid}  \n**Score:** {display_score:.4f}")

                    # Resolve local image path for display
                    # Assuming image_path in payload is relative to IMAGE_DIR or an absolute path
                    img_display_path = path if os.path.exists(path) else os.path.join(IMAGE_DIR, os.path.basename(path))
                    if os.path.exists(img_display_path):
                        try:
                            st.image(Image.open(img_display_path), use_container_width=True)
                        except Exception as img_e:
                            st.warning(f"Could not display image {os.path.basename(path)}: {img_e}")
                    else:
                        st.warning(f"Image file not found for ID {pid}")
    # Corrected indentation for the final else block
    else:
        st.info("Configure your search query and click **Search** in the sidebar.")

else:
    # Message to display if torch import failed or models didn't load
    if torch is None:
         st.error("Failed to load PyTorch. Please ensure it is installed correctly.")
    else:
        st.error("Application resources failed to initialize. Please check error messages above.")