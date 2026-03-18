# embed_data.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import warnings # Import warnings for non-finite embeddings

from constants import DATASET_TSV_PATH, EMBEDDINGS_NPZ_PATH, DEVICE, MODEL_ID # Import from constants
from embed_utils import get_siglip_models_and_processor

def embed_dataset_in_batches(dataset_df: pd.DataFrame, output_path: str, batch_size: int = 32, print_dimensions: bool = False):
    """
    1) Loads a Pandas DataFrame of (product_id, image_path, description)
    2) Batches through SigLIP-2 NaFlex vision & text towers
    3) Normalizes and saves image+text embeddings + metadata to an NPZ
    """
    if dataset_df.empty:
        print("[Error] Input DataFrame is empty.")
        return

    # Ensure necessary columns exist
    if not all(col in dataset_df.columns for col in ['product_id', 'image_path', 'description']):
        print("[Error] Input DataFrame must contain 'product_id', 'image_path', and 'description' columns.")
        return

    # ── 1) Load models & processor ──────────────────────────────────────────────
    processor, vision_model, text_model = get_siglip_models_and_processor(device=DEVICE)

    all_ids, all_paths = [], []
    img_embs_list, txt_embs_list = [], []

    print(f"Embedding {len(dataset_df)} items in batches of {batch_size}…")
    for start in tqdm(range(0, len(dataset_df), batch_size), desc="Batches"):
        batch = dataset_df.iloc[start : start + batch_size]
        pil_images, texts, ids = [], [], []
        batch_image_paths = [] # Store image paths for the current batch

        # 2a) Load images & collect texts/ids
        for pid, img_path, desc in zip(batch.product_id, batch.image_path, batch.description):
            # Ensure image path exists before trying to open
            if not os.path.exists(img_path):
                 # print(f"[Warning] Image file not found: {img_path}. Skipping product ID: {pid}.")
                 continue # Skip if image is missing

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[Warning] Could not open {img_path}: {e}. Skipping product ID: {pid}.")
                continue # Skip if image is corrupt

            pil_images.append(img)
            texts.append(str(desc) if pd.notna(desc) else "") # Handle potential missing/NaN descriptions
            ids.append(pid)
            batch_image_paths.append(img_path)


        if not pil_images:
            # print(f"[Warning] No valid images found in batch starting at index {start}. Skipping batch.")
            continue  # nothing to embed this batch

        # 2b) IMAGE EMBEDDING with dynamic padding (NaFlex)
        img_inputs = processor(
            images=pil_images,
            return_tensors="pt",
            padding=True # NaFlex uses dynamic padding per batch
        ).to(DEVICE)

        pv  = img_inputs.pixel_values
        pam = img_inputs.pixel_attention_mask
        ss  = img_inputs.spatial_shapes

        with torch.no_grad():
            vision_model.eval() # Ensure model is in evaluation mode
            img_out = vision_model(
                pixel_values=pv,
                pixel_attention_mask=pam,
                spatial_shapes=ss
            )
        img_embs = img_out.pooler_output.cpu().numpy()  # (B, D)

        # 2c) TEXT EMBEDDING with padding and truncation
        txt_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True, # Pad to the longest sequence in the batch
            truncation=True, # Truncate to model's max length (default is typically 77 or similar)
            return_attention_mask=True
        ).to(DEVICE)

        input_ids      = txt_inputs["input_ids"]
        attention_mask = txt_inputs["attention_mask"]

        with torch.no_grad():
            text_model.eval() # Ensure model is in evaluation mode
            txt_out = text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        txt_embs = txt_out.pooler_output.cpu().numpy()  # (B, D)

        # 2d) Accumulate (only append data for items that were successfully processed)
        if len(ids) == len(pil_images) == len(texts) == len(img_embs) == len(txt_embs):
            all_ids.extend(ids)
            all_paths.extend(batch_image_paths)
            img_embs_list.append(img_embs)
            txt_embs_list.append(txt_embs)
        else:
            print(f"[Warning] Mismatch in successful item count within batch starting at index {start}. "
                  f"Processed: IDs={len(ids)}, Images={len(pil_images)}, Texts={len(texts)}, "
                  f"ImgEmbs={len(img_embs)}, TxtEmbs={len(txt_embs)}. Skipping batch accumulation.")


    # ── 3) Final checks ──────────────────────────────────────────────────────────
    if not all_ids:
        print("[Error] No embeddings were generated; check your input data and image paths.")
        return

    # ── 4) Stack, normalize, and save ───────────────────────────────────────────
    try:
        img_matrix = np.vstack(img_embs_list)  # (N, D)
        txt_matrix = np.vstack(txt_embs_list)  # (N, D)
    except ValueError as e:
        print(f"[Error] Could not stack embeddings: {e}. This might indicate inconsistent embedding sizes or empty lists after filtering.")
        print(f"Lengths after filtering: ids={len(all_ids)}, paths={len(all_paths)}, img_batches={len(img_embs_list)}, txt_batches={len(txt_embs_list)}")
        return

    # Print dimensions if requested
    if print_dimensions:
        print("\n--- Embedding Dimensions ---")
        print(f"Total items embedded: {len(all_ids)}")
        print(f"Image embeddings shape: {img_matrix.shape}")
        print(f"Text embeddings shape: {txt_matrix.shape}")
        # The embedding dimension 'D' should be the same for both image and text
        if img_matrix.shape[1] != txt_matrix.shape[1]:
            print("[Error] Image and Text embedding dimensions mismatch!")
        print("----------------------------")


    # L2 normalize for cosine similarity
    img_norms = np.linalg.norm(img_matrix, axis=1, keepdims=True)
    txt_norms = np.linalg.norm(txt_matrix, axis=1, keepdims=True)

    # Avoid division by zero: where norm is 0, keep the embedding as 0
    img_matrix = np.divide(img_matrix, img_norms, out=np.zeros_like(img_matrix), where=img_norms!=0)
    txt_matrix = np.divide(txt_matrix, txt_norms, out=np.zeros_like(txt_matrix), where=txt_norms!=0)

    # Ensure embeddings are finite after normalization
    if not np.all(np.isfinite(img_matrix)):
        warnings.warn("[Warning] Non-finite image embeddings found after normalization. Replacing with 0.")
        img_matrix[~np.isfinite(img_matrix)] = 0

    if not np.all(np.isfinite(txt_matrix)):
        warnings.warn("[Warning] Non-finite text embeddings found after normalization. Replacing with 0.")
        txt_matrix[~np.isfinite(txt_matrix)] = 0


    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    np.savez(
        output_path,
        product_ids      = np.array(all_ids, dtype=object),
        image_paths      = np.array(all_paths, dtype=object),
        image_embeddings = img_matrix,
        text_embeddings  = txt_matrix
    )
    print(f"✅ Saved embeddings for {len(all_ids)} items to {output_path}")


# --- Script Entry Point ---
if __name__ == "__main__":

    # --- Configuration ---
    # Toggle this boolean to switch between TEST and FULL run
    IS_TEST_RUN = False
    TEST_LIMIT = 10 # Number of rows to process in test mode

    # --- Test Mode Logic ---
    if IS_TEST_RUN:
        print(f"--- Running Embedding in TEST Mode (first {TEST_LIMIT} rows) ---")
        test_output_path = "temp_test_embeddings.npz" # Temporary file for test embeddings

        try:
            # Load only the first TEST_LIMIT rows from your existing TSV
            print(f"Loading test data from the first {TEST_LIMIT} rows of {DATASET_TSV_PATH}...")
            test_df = pd.read_csv(DATASET_TSV_PATH, sep='\t', nrows=TEST_LIMIT)

            if test_df.empty:
                 print(f"[Error] No data loaded from the first {TEST_LIMIT} rows of {DATASET_TSV_PATH}. "
                       "Ensure the file exists and has at least this many rows.")
            else:
                 embed_dataset_in_batches(
                     dataset_df=test_df,
                     output_path=test_output_path,
                     batch_size=5, # Smaller batch size for test
                     print_dimensions=True # Print dimensions in test mode
                 )
                 print(f"Test embeddings saved to {test_output_path}")

        except FileNotFoundError:
            print(f"[Error] Your dataset TSV file not found at {DATASET_TSV_PATH}. "
                  "Please update DATASET_TSV_PATH in constants.py to point to your existing file.")
        except Exception as e:
            print(f"[Error] An unexpected error occurred during test embedding: {e}")


    # --- Full Dataset Mode Logic ---
    else: # IS_TEST_RUN is False
        print("\n--- Running Embedding on FULL Dataset ---")
        # Load the entire dataset from your existing TSV
        try:
            print(f"Loading full dataset from {DATASET_TSV_PATH}...")
            full_df = pd.read_csv(DATASET_TSV_PATH, sep='\t')

            if full_df.empty:
                print(f"[Error] Full dataset is empty at {DATASET_TSV_PATH}. Please check your file.")
            else:
                 embed_dataset_in_batches(
                     dataset_df=full_df,
                     output_path=EMBEDDINGS_NPZ_PATH,
                     batch_size=32, # Use a larger batch size for full run if memory allows
                     print_dimensions=False # No need to print dimensions for the full run
                 )
                 print(f"Full dataset embeddings saved to {EMBEDDINGS_NPZ_PATH}")

        except FileNotFoundError:
            print(f"[Error] Your dataset TSV file not found at {DATASET_TSV_PATH}. "
                  "Please update DATASET_TSV_PATH in constants.py to point to your existing file.")
        except Exception as e:
            print(f"[Error] An unexpected error occurred during full dataset embedding: {e}")