#!/usr/bin/env python3
import os
import random
import pathlib
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# 1. Setup HEIC support
register_heif_opener()

# ================= CONFIG =================
SOURCE_DIR = "..." # CHANGE THIS to your actual source
DEST_DIR = "training_thumbs"
IMG_SIZE = (160, 160)
MAX_SAMPLES = 8000  # Change for more/less data

pathlib.Path(DEST_DIR).mkdir(parents=True, exist_ok=True)

def smart_resize_crop(img, size):
    """
    Resizes the image to fill the target size (maintaining aspect ratio),
    then crops the center.
    Result: A full-frame square image with NO BLACK BARS.
    """
    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

def prepare_data():
    all_files = []
    print(f"Scanning {SOURCE_DIR}...")
    
    # scan for images
    for root, _, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.heic', '.png')):
                all_files.append(os.path.join(root, f))
    
    print(f"Found {len(all_files)} images.")
    
    # Randomly select samples
    selected = random.sample(all_files, min(MAX_SAMPLES, len(all_files)))
    
    success_count = 0
    print("Processing images (Removing bars, fixing orientation)...")
    
    for i, path in enumerate(selected):
        try:
            with Image.open(path) as img:
                # 1. FORCE UPRIGHT (Critical!)
                img = ImageOps.exif_transpose(img)
                
                # 2. Convert to RGB (Drop Alpha/Grayscale)
                img = img.convert("RGB")

                thumb = smart_resize_crop(img, IMG_SIZE)
                
                # 4. Save
                thumb.save(f"{DEST_DIR}/img_{i}.jpg", "JPEG", quality=90)
                success_count += 1
                
                if i % 500 == 0:
                    print(f"   Processed {i}...")
        except Exception as e:
            # Skip corrupted files
            continue

    print(f"Done! Created {success_count} clean training thumbnails in '{DEST_DIR}'.")

if __name__ == "__main__":
    prepare_data()