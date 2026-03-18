import os
import json
import csv
import html
import pandas as pd
from random import shuffle
from bs4 import BeautifulSoup
from tqdm import tqdm
from constants import IMAGE_DIR, JSON_DIR, DATASET_TSV_PATH



def clean_html(raw_html: str) -> str:
    """Clean HTML content and return plain text."""
    if not raw_html:
        return ""
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    return html.unescape(text).replace("\n", " ").strip()

def build_text(data):
    """Build a concise and meaningful product description from JSON data."""
    # Extract basic product information
    name = data.get("productDisplayName", "").strip()
    brand = data.get("brandName", "").strip()
    gender = data.get("gender", "").strip()
    base_color = data.get("baseColour", "").strip()

    # Get article attributes
    article_attrs = data.get("articleAttributes", {})

    # Extract key attributes with better phrasing
    fit = article_attrs.get("Fit", "").strip().lower()
    pattern = article_attrs.get("Pattern", "").strip().lower()
    sleeve_length = article_attrs.get("Sleeve Length", "").strip().lower()
    neck = article_attrs.get("Neck", "").strip().lower()
    fabric = article_attrs.get("Fabric", "").strip().lower()
    shape = article_attrs.get("Shape", "").strip().lower() # Added Shape

    description_parts = []

    # Start with the core product identity
    core_identity = []
    if name:
        core_identity.append(name)
    if gender and gender.lower() != 'unisex': # Avoid redundant 'unisex for unisex'
         core_identity.append(f"for {gender}")
    if base_color:
        core_identity.append(f"in {base_color}")

    if core_identity:
        description_parts.append(" ".join(core_identity))

    # Add key visual attributes to the main description
    visual_attributes = []
    if pattern and pattern != "solid":
         visual_attributes.append(f"a {pattern} pattern")
    if shape:
        visual_attributes.append(f"a {shape} shape")
    if fit:
        visual_attributes.append(f"a {fit} fit")
    if sleeve_length:
        visual_attributes.append(sleeve_length) # Already lowercased

    if visual_attributes:
        description_parts.append("featuring " + ", ".join(visual_attributes))

    # Add fabric information
    if fabric:
        description_parts.append(f"made of {fabric}")

    # Add neck information
    if neck:
         if description_parts: # Add as a separate detail if there's a preceding part
             description_parts.append(f"with a {neck}")
         else: # If no other details, start with the neck
             description_parts.append(f"{name or 'Garment'} with a {neck}")


    # Add usage information if available
    usage = data.get("usage", "").strip()
    if usage and usage.lower() not in " ".join(description_parts).lower():
        description_parts.append(f"ideal for {usage.lower()}")

    # Combine all parts into a sentence
    final_desc = ". ".join(part for part in description_parts if part) # Filter out empty parts

    if final_desc:
        final_desc += "." # Add a period at the end

    # Add brand information if not already implicitly included or if it makes sense
    if brand and brand.lower() not in final_desc.lower():
         if final_desc:
              final_desc = f"{final_desc} By {brand}."
         else: # If only brand is available
             final_desc = f"A product by {brand}."


    # Clean up any extra spaces and return
    return final_desc.replace("..", ".").strip() # Handle potential double periods

def create_dataset_tsv(test_limit=None):
    """Create a TSV dataset with product ID, image path, and improved descriptions."""
    records = []
    # Create dummy JSON files for testing if not using your constants
    if not os.listdir(JSON_DIR):
        dummy_data = [
            {"id": 1, "productDisplayName": "Classic T-Shirt", "brandName": "BrandX", "gender": "Men", "baseColour": "Blue", "articleAttributes": {"Fit": "Regular Fit", "Pattern": "Solid", "Sleeve Length": "Short Sleeves", "Neck": "Round Neck", "Fabric": "Cotton"}},
            {"id": 2, "productDisplayName": "Floral Dress", "brandName": "BrandY", "gender": "Women", "baseColour": "Red", "articleAttributes": {"Pattern": "Printed", "Shape": "Fit and Flare", "Neck": "Round Neck", "Fabric": "Blended"}, "usage": "Casual wear"},
            {"id": 3, "productDisplayName": "Striped Polo", "brandName": "BrandX", "gender": "Men", "baseColour": "Green", "articleAttributes": {"Fit": "Regular Fit", "Pattern": "Striped", "Sleeve Length": "Short Sleeves", "Neck": "Polo Collar", "Fabric": "Cotton"}},
            {"id": 4, "productDisplayName": "Basic Jeans", "brandName": "BrandZ", "gender": "Unisex", "baseColour": "Black", "articleAttributes": {"Fit": "Slim Fit", "Pattern": "Solid"}},
            {"id": 5, "productDisplayName": "Simple Top", "brandName": "BrandY", "gender": "Women", "baseColour": "White", "articleAttributes": {}} # No attributes
        ]
        for item in dummy_data:
            with open(os.path.join(JSON_DIR, f"{item['id']}.json"), "w") as f:
                json.dump({"data": item}, f) # Wrap in "data" key as per your original code structure


    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

    if test_limit is not None:
        json_files = json_files[:test_limit]
        print(f"Running in test mode: processing {len(json_files)} files")

    for idx, fname in enumerate(tqdm(json_files, desc="Building TSV")):
        path = os.path.join(JSON_DIR, fname)
        try:
            with open(path, "r", encoding="utf8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue  # skip corrupt JSON

        data = rec.get("data") or rec
        pid = str(data.get("id") or data.get("product_id") or os.path.splitext(fname)[0])
        img_file = os.path.join(IMAGE_DIR, f"{pid}.jpg")

        # Create a dummy image file for testing if it doesn't exist
        if not os.path.isfile(img_file):
             with open(img_file, 'w') as f:
                 f.write("dummy image content")


        text = build_text(data)
        if not text:
            continue

        records.append({
            "product_id": pid,
            "image_path": img_file,
            "description": text
        })

    df = pd.DataFrame(records)
    df.to_csv(DATASET_TSV_PATH, sep="\t", index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} rows to {DATASET_TSV_PATH}")
    print(df.head())

if __name__ == "__main__":
    create_dataset_tsv(test_limit=None) # Set a small limit for testing
    # create_dataset_tsv(test_limit=None) # Set to None or 0 for full dataset