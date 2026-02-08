import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

# ================= CONFIG =================
SOURCE_DIR = "..."
DATASET_DIR = "live_verification_dataset"
IMG_SIZE = (224, 224)
FRAMES_PER_VIDEO = 7 
# ========================================

def extract_frames(video_path, num_frames=7):
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return frames

    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames

def load_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        return np.array(img)
    except:
        return None

def save_thumb(arr, path):
    img = Image.fromarray(arr)
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img.save(path, "JPEG", quality=80)

if __name__ == "__main__":
    src = Path(SOURCE_DIR)

    out_imgs = Path(DATASET_DIR) / "stills"
    out_frames = Path(DATASET_DIR) / "frames"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_frames.mkdir(parents=True, exist_ok=True)

    pairs = {}
    for f in src.rglob("*"):
        if f.name.startswith("."):
            continue
        pairs.setdefault(f.stem, {})
        if f.suffix.lower() in [".jpg", ".jpeg", ".heic"]:
            pairs[f.stem]["img"] = f
        elif f.suffix.lower() in [".mov", ".mp4"]:
            pairs[f.stem]["vid"] = f

    valid = [p for p in pairs.values() if "img" in p and "vid" in p]
    print(f"Found {len(valid)} Live Photo pairs")

    for p in valid:
        stem = p["img"].stem
        still_out = out_imgs / f"{stem}.jpg"
        if still_out.exists():
            continue

        img = load_image(p["img"])
        if img is None:
            continue
        save_thumb(img, still_out)

        frames = extract_frames(p["vid"], FRAMES_PER_VIDEO)
        for i, frame in enumerate(frames):
            save_thumb(frame, out_frames / f"{stem}__f{i}.jpg")

    print("Dataset creation complete.")
