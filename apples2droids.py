#!/usr/bin/env python3
"""
ROBUST TWO-STAGE APPROACH:
1. Let exiftool do its thing completely
2. Save the file atomically 
4. Use AI-2 to verify the image+video pair if selected
4. Use AI-1 to check if the saved file matches the original (orientation)

All new Motion Photos are saved to output, along with all unchanged files.
"""

from ML_algorithm_2.custom_layers import AbsDiff, ReduceMin, ReduceMean

import os
import shutil
import subprocess
import threading
import queue
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import tempfile
from pathlib import Path
from multiprocessing import cpu_count, freeze_support
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= IMAGE HANDLING SETUP =================
from PIL import Image, ImageOps, ImageTk
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("WARNING: 'pillow-heif' not installed. AI may fail on HEIC files.")
    print("Run: pip install pillow-heif")

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
    HAS_PT = True
except Exception:
    HAS_PT = False

model_pt = None

# ================= TENSORFLOW & GPU SETUP =================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

HAS_TF = False
try:
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.utils import register_keras_serializable

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Acceleration Enabled: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU Config Error: {e}. Continuing on CPU/Fallback.")
    else:
        print("No GPU detected. Running on CPU (this is fine).")

    HAS_TF = True
except ImportError:
    print("TensorFlow not found.")

# ================= CONSTANTS =================
MODEL_PATH_TF = "ML_algorithm_1/relative_rotation_alignment_model.keras"
VERIFIER_MODEL_PATH = "ML_algorithm_2/live_photo_verifier_model.keras"
IMG_SIZE = (160, 160)
VERIFIER_IMG_SIZE = (224, 224)
VERIFIER_SAMPLE_POS = [0.25, 0.5, 0.75]
DEFAULT_PER_WORKER_MEM_MB = 700
MIN_FREE_MEM_MB = 1000
MIN_MEM_FOR_MODEL = 2000


def get_available_memory_mb() -> int:
    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            return int(vm.available / 1024 / 1024)
        with open('/proc/meminfo', 'r') as f:
            lines = f.read().splitlines()
        mem = {}
        for line in lines:
            parts = line.split(':')
            if len(parts) < 2: continue
            key = parts[0].strip()
            val = parts[1].strip().split()[0]
            mem[key] = int(val)
        if 'MemAvailable' in mem:
            return int(mem['MemAvailable'] / 1024)
    except Exception:
        pass
    return None


def safe_worker_count(max_hint: int, per_worker_mb: int = DEFAULT_PER_WORKER_MEM_MB) -> int:
    env = os.getenv('PER_WORKER_MEM_MB')
    if env:
        try:
            per_worker_mb = int(env)
        except Exception:
            pass

    avail = get_available_memory_mb()
    if avail is None:
        return max(1, max_hint)

    usable = max(0, avail - MIN_FREE_MEM_MB)
    if usable <= 0:
        return 1

    by_mem = max(1, usable // per_worker_mb)
    return max(1, min(max_hint, by_mem))

# ================= UTILS =================
def run_silent(cmd):
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# ================= AI MODEL LAYERS =================

@register_keras_serializable()
class RotateLayer(Layer):
    def __init__(self, k=0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
    def call(self, inputs):
        return tf.image.rot90(inputs, k=self.k)
    def get_config(self):
        cfg = super().get_config()
        cfg["k"] = self.k
        return cfg
    
@register_keras_serializable()
class AbsLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.abs(inputs)

model_tf = None
inference_worker = None

class InferenceWorker:
    def __init__(self, model, max_batch=8, timeout=0.04, confidence_thresh=0.90):
        self.model = model
        self.max_batch = max_batch
        self.timeout = timeout
        self.confidence_thresh = confidence_thresh
        try:
            mem = get_available_memory_mb()
            if mem is not None:
                if mem < 3000:
                    self.max_batch = min(self.max_batch, 2)
                elif mem < 6000:
                    self.max_batch = min(self.max_batch, 4)
        except Exception:
            pass

        self.tasks = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            batch = []
            try:
                task = self.tasks.get(timeout=self.timeout)
                batch.append(task)
            except queue.Empty:
                continue

            while len(batch) < self.max_batch:
                try:
                    task = self.tasks.get_nowait()
                    batch.append(task)
                except queue.Empty:
                    break

            try:
                refs = np.concatenate([t.ref for t in batch], axis=0)
                checks = np.concatenate([t.check for t in batch], axis=0)
                preds = self.model.predict([refs, checks], verbose=0)
                for i, t in enumerate(batch):
                    probs = preds[i]
                    cls = int(np.argmax(probs))
                    max_prob = float(np.max(probs))

                    if max_prob < self.confidence_thresh:
                        try:
                            check_arr = np.squeeze(t.check)
                            ref_arr = np.squeeze(t.ref)
                            diffs = [np.mean(np.abs(ref_arr - np.rot90(check_arr, k=k))) for k in range(4)]
                            det_cls = int(np.argmin(diffs))
                            t.result = det_cls
                        except Exception:
                            t.result = cls
                    else:
                        t.result = cls

                    t.event.set()
            except Exception:
                for t in batch:
                    try:
                        check_arr = np.squeeze(t.check)
                        ref_arr = np.squeeze(t.ref)
                        diffs = [np.mean(np.abs(ref_arr - np.rot90(check_arr, k=k))) for k in range(4)]
                        t.result = int(np.argmin(diffs))
                    except Exception:
                        t.result = 0
                    t.event.set()

    def predict(self, ref, check):
        if self._stop.is_set():
            # If worker is stopped, return default to avoid hang
            return 0
        task = type('T', (), {})()
        task.ref = ref
        task.check = check
        task.event = threading.Event()
        task.result = None
        self.tasks.put(task)
        task.event.wait()
        return int(task.result) if task.result is not None else 0

    def stop(self):
        self._stop.set()
        try:
            while not self.tasks.empty():
                self.tasks.get_nowait()
        except Exception:
            pass
        self._thread.join(timeout=1.0)


def load_model_tf(logger):
    global model_tf, inference_worker
    if not HAS_TF:
        logger("TensorFlow not available. Attempting PyTorch fallback.")
        if HAS_PT:
            load_model_pt(logger)
        else:
            logger("Neither TensorFlow nor PyTorch available for AI inference.")
        return

    mem_avail = get_available_memory_mb()
    if mem_avail is not None and mem_avail < MIN_MEM_FOR_MODEL:
        logger(f"Warning: low available memory ({mem_avail} MB). Attempting to load the AI model nonetheless.")

    if not os.path.exists(MODEL_PATH_TF):
        logger(f"Model file '{MODEL_PATH_TF}' not found. AI skipped.")
        return

    try:
        logger("Loading AI Model...")
        try:
            intra = max(1, min(4, cpu_count() // 2))
            inter = 1
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            tf.config.threading.set_inter_op_parallelism_threads(inter)
            logger(f"TF threads: intra={intra}, inter={inter}")
        except Exception:
            pass

        model_tf = tf.keras.models.load_model(
            MODEL_PATH_TF,
            custom_objects={"RotateLayer": RotateLayer, "AbsLayer": AbsLayer}
        )
        inference_worker = InferenceWorker(model_tf)
        logger("AI Model Ready.")
    except Exception as e:
        logger(f"Model Load Failed: {e}")
        if HAS_PT:
            try:
                load_model_pt(logger)
            except Exception as e2:
                logger(f"PyTorch fallback also failed: {e2}")


def safe_load_image_with_exif(path):
    """Load image and apply EXIF orientation (for viewing correctness)."""
    for attempt in range(3):
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                time.sleep(0.2)
                continue

            with Image.open(path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert('RGB')
                pil_img = pil_img.resize(IMG_SIZE, Image.BICUBIC)
                arr = (np.asarray(pil_img, dtype=np.float32) / 255.0)

            return np.expand_dims(arr, axis=0)
        except Exception:
            time.sleep(0.2)
    return None


def load_model_pt(logger):
    global model_pt
    if not HAS_PT:
        logger("PyTorch not available on this system.")
        return
    path = "relative_rotation_alignment_model_pt.pth"
    if not os.path.exists(path):
        logger(f"PyTorch model file '{path}' not found.")
        return
    try:
        import train_pt as train_pt_module
        model = train_pt_module.RotationClassifier()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        model_pt = model
        logger("PyTorch model ready.")
    except Exception as e:
        logger(f"Failed to load PyTorch model: {e}")


# ================= VERIFIER MODEL (AI ALGO 2) =================
verifier_model = None

def get_video_duration(path: str):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ], stderr=subprocess.DEVNULL)
        return float(out.decode().strip())
    except Exception:
        return None


def sample_video_frames(video_path: str, positions, needed_count: int):
    """Extract frames from video at relative positions (0.0-1.0). Returns list of file paths."""
    dur = get_video_duration(video_path)
    if dur is None:
        return []

    tmp_files = []
    # If requested positions fewer than needed_count, we'll duplicate later
    for p in positions:
        t = max(0.1, min(dur - 0.1, dur * float(p)))
        tfp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tfp.close()
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            tfp.name
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            tmp_files.append(tfp.name)
        except Exception:
            try:
                os.unlink(tfp.name)
            except Exception:
                pass

    # If we need more frames, duplicate nearest frames until count reached
    if len(tmp_files) > 0 and len(tmp_files) < needed_count:
        i = 0
        while len(tmp_files) < needed_count:
            tmp_files.append(tmp_files[i % len(tmp_files)])
            i += 1

    return tmp_files


def load_verifier_model(logger):
    global verifier_model
    if verifier_model is not None:
        return
    try:
        if not HAS_TF:
            logger("Verifier: TensorFlow not available, verifier skipped.")
            return
        if not os.path.exists(VERIFIER_MODEL_PATH):
            logger(f"Verifier model '{VERIFIER_MODEL_PATH}' not found. Skipping AI-algo-2.")
            return
        logger("Loading verifier model (AI algorithm 2)...")
        verifier_model = tf.keras.models.load_model(VERIFIER_MODEL_PATH)
        logger("Verifier model ready.")
    except Exception as e:
        logger(f"Failed to load verifier model: {e}")


def verify_pair(still_path: str, video_path: str, logger, sample_positions=VERIFIER_SAMPLE_POS):
    """Return (status, confidence) tuple where:
    - status: "match" (prob >= 0.70), "uncertain" (0.40 <= prob < 0.70), or "mismatch" (prob < 0.40)
    - confidence: 0.0-1.0 probability
    """
    global verifier_model
    if verifier_model is None:
        return ("verifier_unavailable", 0.0)

    try:
        # Determine model expected frames length
        try:
            frames_input_shape = verifier_model.input[1].shape
            expected_frames = int(frames_input_shape[1]) if frames_input_shape[1] is not None else len(sample_positions)
        except Exception:
            expected_frames = len(sample_positions)

        tmp_frames = sample_video_frames(video_path, sample_positions, expected_frames)
        if not tmp_frames:
            logger("Verifier: could not sample frames from video; assuming match.")
            return ("match", 1.0)

        # Load still
        still = load_img(still_path, target_size=VERIFIER_IMG_SIZE)
        still = img_to_array(still) / 255.0
        still = np.expand_dims(still, axis=0)

        frame_imgs = []
        for f in tmp_frames[:expected_frames]:
            img = load_img(f, target_size=VERIFIER_IMG_SIZE)
            img = img_to_array(img) / 255.0
            frame_imgs.append(img)

        # If we have fewer than expected, duplicate
        while len(frame_imgs) < expected_frames:
            frame_imgs.append(frame_imgs[-1])

        frames_arr = np.expand_dims(np.stack(frame_imgs, axis=0), axis=0)

        preds = verifier_model.predict([still, frames_arr], verbose=0)
        # model outputs a single sigmoid probability per sample
        prob = float(preds.reshape(-1)[0])

        # cleanup temporary files
        for f in set(tmp_frames):
            try:
                os.unlink(f)
            except Exception:
                pass

        # Classify into three tiers
        if prob >= 0.70:
            status = "match"
        elif prob < 0.40:
            status = "mismatch"
        else:
            status = "uncertain"

        return (status, prob)
    except Exception as e:
        logger(f"Verifier error: {e} — allowing pair by default")
        return ("match", 1.0)



def get_correction_rotation_class(ref_path, check_path):
    """
    Returns: rotation class (0-3) where:
    - 0 = no change needed
    - 1 = rotate 90° CCW needed
    - 2 = rotate 180° needed  
    - 3 = rotate 270° CCW needed
    """
    # Load both images WITH EXIF correction
    arr_ref = safe_load_image_with_exif(ref_path)
    if arr_ref is None:
        print(f"   [AI Fail] Could not read reference: {os.path.basename(ref_path)}")
        return 0

    arr_check = safe_load_image_with_exif(check_path)
    if arr_check is None:
        print(f"   [AI Fail] Could not read check file: {os.path.basename(check_path)}")
        return 0

    cls = None
    # Try TF inference first
    try:
        if model_tf is not None:
            if inference_worker is not None:
                cls = inference_worker.predict(arr_ref, arr_check)
            else:
                preds = model_tf.predict([arr_ref, arr_check], verbose=0)
                cls = int(np.argmax(preds))
    except Exception as e:
        print(f"   [AI Error] TF inference failed: {e}")

    # PyTorch fallback
    if cls is None and HAS_PT and model_pt is None:
        try:
            load_model_pt(lambda m: print(m))
        except Exception:
            pass

    if cls is None and model_pt is not None:
        try:
            ref = torch.from_numpy(np.transpose(arr_ref[0], (2, 0, 1))).unsqueeze(0).float()
            chk = torch.from_numpy(np.transpose(arr_check[0], (2, 0, 1))).unsqueeze(0).float()
            device = next(model_pt.parameters()).device
            ref = ref.to(device)
            chk = chk.to(device)
            model_pt.eval()
            with torch.no_grad():
                out = model_pt(ref, chk)
                cls = int(torch.argmax(out, dim=1).item())
        except Exception as e:
            print(f"   [AI Error] PT inference failed: {e}")

    # Deterministic fallback
    if cls is None:
        try:
            ref_arr = np.squeeze(arr_ref)
            check_arr = np.squeeze(arr_check)
            diffs = [np.mean(np.abs(ref_arr - np.rot90(check_arr, k=k))) for k in range(4)]
            cls = int(np.argmin(diffs))
        except Exception:
            cls = 0

    return int(cls) if cls is not None else 0


def rotate_pixels_in_place(filepath, rotation_class, logger):
    """
    rotation_class:
    1 = 90° CCW
    2 = 180°
    3 = 270° CCW
    """
    if rotation_class == 0:
        return True

    angle_map = {
        1: 90,
        2: 180,
        3: 270
    }

    angle = angle_map.get(rotation_class)
    if angle is None:
        return False

    try:
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = img.rotate(angle, expand=True)
            img.save(filepath, "JPEG", quality=95, subsampling=0)

        # Now FORCE orientation to 1 (safe)
        run_silent([
            "exiftool", "-overwrite_original",
            "-Orientation=1",
            "-IFD0:Orientation=1",
            "-EXIF:Orientation=1",
            filepath
        ])

        logger(f"   ✓ Pixels rotated {angle}° and Orientation reset to 1")
        return True

    except Exception as e:
        logger(f"   ⚠ Pixel rotation failed: {e}")
        return False


# ================= PROCESSING LOGIC =================

def convert_image(src: Path, dst: Path):
    if shutil.which("heif-convert"):
        if run_silent(["heif-convert", str(src), str(dst)]):
            return True
    return run_silent(["ffmpeg", "-y", "-i", str(src), str(dst)])

def probe_video_codec(path: Path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ], stderr=subprocess.DEVNULL)
        codec = out.decode().strip().splitlines()[0] if out else None
        return codec
    except Exception:
        return None


def transcode_or_copy_video(src: Path, dst: Path, logger) -> bool:
    codec = probe_video_codec(src)
    if codec == "h264":
        logger(f"   Video codec h264 detected - remuxing (fast)")
        return run_silent(["ffmpeg", "-y", "-i", str(src), "-c:v", "copy", "-an", str(dst)])

    logger("   Transcoding video (may take a moment)")
    return run_silent([
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-an", "-threads", "1", str(dst)
    ])


def process_live_pair(img_path: Path, vid_path: Path, out_dir: Path, logger, ai2_enabled=False, ai2_auto_pass=True, app=None):
    """
    ROBUST TWO-STAGE PROCESS:
    1. Convert, merge, apply metadata (let exiftool do its thing completely)
    2. AI check the final result
    3. If wrong, fix orientation via exiftool
    """
    tmp_jpg = out_dir / (img_path.stem + ".tmp.jpg")
    tmp_mp4 = out_dir / (img_path.stem + ".tmp.mp4")
    final_path = out_dir / (img_path.stem + ".jpg")

    try:
        # If AI algorithm 2 is enabled, run verifier BEFORE expensive processing.
        if ai2_enabled:
            try:
                logger("   Verifying still/video match (AI-2)...")
                status, confidence = verify_pair(str(img_path), str(vid_path), logger)

                if status == "verifier_unavailable":
                    logger("   Verifier model not loaded — skipping AI-2 pre-check.")

                elif status == "mismatch":
                    # Strong mismatch: ALWAYS skip merge
                    logger(f"   ⚠ MISMATCH DETECTED (<40%): {img_path.name} + {vid_path.name}")
                    logger(f"   ⚠ NOT merging these files (strong semantic mismatch)")
                    tgt_img = out_dir / img_path.name
                    tgt_vid = out_dir / vid_path.name
                    if tgt_img.exists(): tgt_img = out_dir / f"{img_path.stem}_{int(time.time())}{img_path.suffix}"
                    if tgt_vid.exists(): tgt_vid = out_dir / f"{vid_path.stem}_{int(time.time())}{vid_path.suffix}"
                    try:
                        shutil.copy2(img_path, tgt_img)
                        shutil.copy2(vid_path, tgt_vid)
                        logger(f"   ✓ Files copied unchanged to output.")
                    except Exception as e:
                        logger(f"   Copy failed: {e}")
                    return

                elif status == "uncertain":
                    # Ambiguity zone: handle based on user setting
                    logger(f"   ⚠ UNCERTAIN (40-70%): {img_path.name} + {vid_path.name}")
                    if ai2_auto_pass:
                        # Auto-pass mode: skip merge
                        logger(f"   ⚠ Auto-skip: not merging due to ambiguity")
                        tgt_img = out_dir / img_path.name
                        tgt_vid = out_dir / vid_path.name
                        if tgt_img.exists(): tgt_img = out_dir / f"{img_path.stem}_{int(time.time())}{img_path.suffix}"
                        if tgt_vid.exists(): tgt_vid = out_dir / f"{vid_path.stem}_{int(time.time())}{vid_path.suffix}"
                        try:
                            shutil.copy2(img_path, tgt_img)
                            shutil.copy2(vid_path, tgt_vid)
                            logger(f"   ✓ Files copied unchanged to output.")
                        except Exception as e:
                            logger(f"   Copy failed: {e}")
                        return
                    else:
                        # Manual verification mode: ask user
                        if app is not None:
                            app.misidentified.append((img_path, vid_path))
                            logger(f"   ✓ Registered for manual verification (will prompt at end)")
                        else:
                            logger("   Manual verification requested but app context missing — skipping pair")
                        return

                else:
                    # Confident match (>= 70%): proceed normally
                    logger(f"   AI-2 match confidence: {confidence:.1%}")
                    logger(f"   ✓ CONFIDENT MATCH (≥70%): {img_path.name} + {vid_path.name} - proceeding with processing.")
            except Exception as e:
                logger(f"   Verifier check failed: {e} — continuing")
        # ========== STAGE 1: Convert, Merge, Metadata ==========
        
        # 1. Convert Image
        if not convert_image(img_path, tmp_jpg):
            logger(f"Conversion failed: {img_path.name}")
            return

        # 2. Transcode video
        success = transcode_or_copy_video(vid_path, tmp_mp4, logger)
        if not success:
            logger(f"Video processing failed: {vid_path.name}")
            return

        # 3. Merge into temporary file
        final_tmp = out_dir / (img_path.stem + ".jpg.part")
        with open(final_tmp, "wb") as f:
            with open(tmp_jpg, "rb") as j: shutil.copyfileobj(j, f)
            with open(tmp_mp4, "rb") as m: shutil.copyfileobj(m, f)

        # 4. Let exiftool copy metadata
        video_offset = final_tmp.stat().st_size - tmp_mp4.stat().st_size
        logger(f"   Copying metadata from original...")
        run_silent([
            "exiftool", "-overwrite_original",
            "-TagsFromFile", str(img_path),
            "-all:all",
            "--Orientation",
            "--Rotation",
            "--RotationMatrix",
            "--IFD0:Orientation",
            "--EXIF:Orientation",
            "-Orientation=1",
            f"-XMP-GCamera:MicroVideoOffset={video_offset}",
            str(final_tmp)
        ])

        # 5. Atomic rename to final filename
        try:
            if final_path.exists():
                timestamp = int(time.time())
                final_path = out_dir / f"{img_path.stem}_{timestamp}.jpg"
            final_tmp.rename(final_path)
            logger(f"Saved (stage 1): {final_path.name}")
        except Exception as e:
            logger(f"Rename failed: {e}")
            return

        # ========== STAGE 2: AI Verification & Fix ==========
        
        # Give file system time to flush
        time.sleep(0.1)
        
        logger(f"   AI checking orientation...")
        try:
            rotation_class = get_correction_rotation_class(str(img_path), str(final_path))
            logger(f"   AI detected rotation class: {rotation_class}")
            
            if rotation_class != 0:
                logger(f"   ⚠ Orientation mismatch detected - fixing (PIXELS)...")
                rotate_pixels_in_place(final_path, rotation_class, logger)
            else:
                logger(f"   ✓ Orientation correct")
                
        except Exception as e:
            logger(f"   AI verification failed: {e} - leaving as-is")

    except Exception as e:
        logger(f"Error {img_path.name}: {e}")
    finally:
        if tmp_jpg.exists(): tmp_jpg.unlink()
        if tmp_mp4.exists(): tmp_mp4.unlink()

def copy_unchanged(src: Path, dst: Path, logger):
    try:
        shutil.copy2(src, dst)
        logger(f"Copied: {src.name}")
    except Exception as e:
        logger(f"Copy Fail: {src.name}")

# ================= GUI APP =================

class Apples2DroidsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apples2Droids")
        self.root.geometry("900x750")

        self.source_dir = tk.StringVar()
        self.dest_dir = tk.StringVar()
        self.msg_queue = queue.Queue()
        self.is_running = False

        ttk.Label(root, text="Apples2Droids", font=("Helvetica", 16, "bold")).pack(pady=5)
        
        cfg = ttk.LabelFrame(root, text="Folders", padding=10)
        cfg.pack(fill="x", padx=10)
        ttk.Entry(cfg, textvariable=self.source_dir, width=60).grid(row=0, column=0)
        ttk.Button(cfg, text="Source", command=self.browse_source).grid(row=0, column=1)
        ttk.Entry(cfg, textvariable=self.dest_dir, width=60).grid(row=1, column=0)
        ttk.Button(cfg, text="Output", command=self.browse_dest).grid(row=1, column=1)

        fmt = ttk.LabelFrame(root, text="Formats", padding=10)
        fmt.pack(fill="x", padx=10, pady=5)
        
        self.use_heic = tk.BooleanVar(value=True)
        self.use_mov = tk.BooleanVar(value=True)
        self.use_jpg = tk.BooleanVar(value=False)
        self.use_mp4 = tk.BooleanVar(value=False)
        self.use_ai2 = tk.BooleanVar(value=False)
        self.ai2_auto_pass = tk.BooleanVar(value=True)
        self.misidentified = []

        ttk.Checkbutton(fmt, text=".heic (Standard)", variable=self.use_heic).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(fmt, text=".mov (Standard)", variable=self.use_mov).grid(row=0, column=1, sticky="w")
        
        lbl_warn = ttk.Label(fmt, text="Only enable below if iPhone settings used 'Most Compatible' or photos are pre-iOS 11", foreground="#d9534f")
        lbl_warn.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10,0))

        ttk.Checkbutton(fmt, text=".jpg (Legacy)", variable=self.use_jpg).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(fmt, text=".mp4 (Legacy)", variable=self.use_mp4).grid(row=2, column=1, sticky="w")

        ai_frame = ttk.LabelFrame(root, text="AI Options", padding=10)
        ai_frame.pack(fill="x", padx=10, pady=(0,5))
        
        # Use GRID for the checkbutton to match the frame's manager if we add more
        ttk.Checkbutton(ai_frame, text="Use AI Algorithm 2 (verify image+video match)", variable=self.use_ai2, command=self._on_ai2_toggle).grid(row=0, column=0, sticky="w")

        self._ai2_modes_frame = ttk.Frame(ai_frame)
        ttk.Radiobutton(
            self._ai2_modes_frame,
            text="Pass all flagged pairs unchanged (all pairs <70% confidence)",
            value=True,
            variable=self.ai2_auto_pass
        ).pack(anchor="w")

        ttk.Radiobutton(
            self._ai2_modes_frame,
            text="Individually verify ambiguous flagged pairs at end (pairs between 40-70% confidence)",
            value=False,
            variable=self.ai2_auto_pass
        ).pack(anchor="w")
        
        self._ai2_modes_frame.grid(row=1, column=0, sticky="w", pady=(6,0))
        self._ai2_modes_frame.grid_remove()

        self.btn_start = ttk.Button(root, text="START PROCESSING", state="disabled", command=self.start_thread)
        self.btn_start.pack(pady=10)

        self.logbox = scrolledtext.ScrolledText(root, height=20, state="disabled")
        self.logbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.progress = ttk.Progressbar(root, mode='indeterminate')
        self.progress.pack(fill="x", padx=10, pady=5)

        self.root.after(100, self.process_queue)

    def _on_ai2_toggle(self):
        if self.use_ai2.get():
            # FIXED: Use grid() instead of pack()
            self._ai2_modes_frame.grid()
        else:
            self._ai2_modes_frame.grid_remove()

    def _extract_video_frames_for_preview(self, vid_path: Path, num_frames=5):
        """Extract multiple frames from video for slideshow preview."""
        frames = []
        temp_files = []
        
        try:
            # Get video duration
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(vid_path)],
                capture_output=True, text=True, check=True
            )
            try:
                duration = float(result.stdout.strip())
            except ValueError:
                duration = 2.0
            
            # Extract frames at evenly spaced intervals
            for i in range(num_frames):
                position = (i / (num_frames - 1)) * duration if num_frames > 1 else 0
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                    temp_files.append(tmp_path)
                    
                # Close file handle so ffmpeg can write to it
                
                subprocess.run([
                    "ffmpeg", "-y", "-ss", f"{position}", "-i", str(vid_path),
                    "-vframes", "1", "-q:v", "2", tmp_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    try:
                        pil_img = Image.open(tmp_path)
                        pil_img.thumbnail((350, 350), Image.Resampling.LANCZOS)
                        frames.append(pil_img)
                    except Exception:
                        pass
                
        except Exception as e:
            print(f"Frame extraction error: {e}")
            return None, temp_files
        
        return frames, temp_files

    def _present_verify_dialog(self, img_path: Path, vid_path: Path):
        """Enhanced verification dialog with side-by-side preview and video frame slideshow."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Verify Live Photo Pair")
        dialog.geometry("1000x750")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {"combine": False}
        temp_files = []
        
        # Title
        ttk.Label(
            dialog, 
            text=f"Possible mismatch detected (40-70% confidence)",
            font=("Helvetica", 12, "bold")
        ).pack(pady=10)
        
        ttk.Label(
            dialog,
            text=f"Image: {img_path.name}\nVideo: {vid_path.name}",
            font=("Helvetica", 10)
        ).pack(pady=5)
        
        # Frame for side-by-side preview
        preview_frame = ttk.Frame(dialog)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Image preview
        img_frame = ttk.LabelFrame(preview_frame, text="Static Image", padding=10)
        img_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        img_label = ttk.Label(img_frame)
        img_label.pack()
        
        try:
            # Load and resize image
            pil_img = Image.open(img_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img.thumbnail((450, 450), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(pil_img)
            img_label.config(image=photo_img)
            img_label.image = photo_img  # Keep reference
        except Exception as e:
            img_label.config(text=f"Cannot preview image:\n{e}")
        
        # Right: Video preview with slideshow
        vid_frame = ttk.LabelFrame(preview_frame, text="Video Frames (auto-playing)", padding=10)
        vid_frame.pack(side="right", fill="both", expand=True, padx=5)
        
        vid_label = ttk.Label(vid_frame, text="Loading preview...")
        vid_label.pack()
        
        frame_index_label = ttk.Label(vid_frame, text="", font=("Helvetica", 8))
        frame_index_label.pack()
        
        # Force GUI update to show "Loading"
        dialog.update()
        
        # Extract frames
        video_frames, temp_files = self._extract_video_frames_for_preview(vid_path, num_frames=5)
        
        slideshow_state = {"index": 0, "running": True, "photo_refs": []}

        if video_frames:
            # Convert PIL images to PhotoImage
            for frame in video_frames:
                photo = ImageTk.PhotoImage(frame)
                slideshow_state["photo_refs"].append(photo)
            
            def update_slideshow():
                if not slideshow_state["running"]:
                    return
                try:
                    if not dialog.winfo_exists():
                        slideshow_state["running"] = False
                        return
                    
                    idx = slideshow_state["index"]
                    photo = slideshow_state["photo_refs"][idx]
                    vid_label.config(image=photo)
                    vid_label.image = photo
                    frame_index_label.config(text=f"Frame {idx + 1} of {len(video_frames)}")
                    
                    # Move to next frame
                    slideshow_state["index"] = (idx + 1) % len(video_frames)
                    
                    # Schedule next update (800ms between frames)
                    dialog.after(800, update_slideshow)
                except Exception:
                    slideshow_state["running"] = False
            
            # Start slideshow
            update_slideshow()
        else:
            vid_label.config(text="Cannot preview video (ffmpeg failed or empty)")
        
        # Question
        ttk.Label(
            dialog,
            text="Do these files belong together as a Live Photo?",
            font=("Helvetica", 11, "bold")
        ).pack(pady=15)
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        def on_yes():
            result["combine"] = True
            slideshow_state["running"] = False
            dialog.destroy()
        
        def on_no():
            result["combine"] = False
            slideshow_state["running"] = False
            dialog.destroy()
        
        def on_close():
            slideshow_state["running"] = False
            result["combine"] = False
            dialog.destroy()
        
        dialog.protocol("WM_DELETE_WINDOW", on_close)
        
        ttk.Button(
            btn_frame, 
            text="✓ YES - Combine them", 
            command=on_yes,
            width=25
        ).pack(side="left", padx=10)
        
        ttk.Button(
            btn_frame, 
            text="✗ NO - Keep separate", 
            command=on_no,
            width=25
        ).pack(side="right", padx=10)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        
        return result["combine"]

    def _ask_user_sync(self, pair):
        # pair is (img_path, vid_path)
        evt = threading.Event()
        result_container = {}

        def _dialog():
            try:
                res = self._present_verify_dialog(pair[0], pair[1])
            except Exception:
                res = False
            result_container['res'] = res
            evt.set()

        self.root.after(10, _dialog)
        evt.wait()
        return bool(result_container.get('res', False))

    def browse_source(self):
        p = filedialog.askdirectory()
        if p:
            self.source_dir.set(p)
            self.check_ready()

    def browse_dest(self):
        p = filedialog.askdirectory()
        if p:
            self.dest_dir.set(p)
            self.check_ready()

    def check_ready(self):
        if self.source_dir.get() and self.dest_dir.get():
            self.btn_start["state"] = "normal"

    def log(self, msg):
        self.logbox.configure(state="normal")
        self.logbox.insert(tk.END, msg + "\n")
        self.logbox.see(tk.END)
        self.logbox.configure(state="disabled")

    def process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def start_thread(self):
        if self.is_running: return
        self.is_running = True
        self.btn_start["state"] = "disabled"
        self.progress.start(10)
        threading.Thread(target=self.run).start()

    def run(self):
        src_root = Path(self.source_dir.get())
        dst_root = Path(self.dest_dir.get())
        
        load_model_tf(lambda m: self.msg_queue.put(m))
        # Load verifier model for AI algorithm 2 (if present)
        load_verifier_model(lambda m: self.msg_queue.put(m))
        self.msg_queue.put(f"Scanning {src_root}...")
        
        img_exts = set()
        if self.use_heic.get(): img_exts.update({'.heic', '.HEIC'})
        if self.use_jpg.get(): img_exts.update({'.jpg', '.JPG', '.jpeg', '.JPEG'})
        
        vid_exts = set()
        if self.use_mov.get(): vid_exts.update({'.mov', '.MOV'})
        if self.use_mp4.get(): vid_exts.update({'.mp4', '.MP4'})

        pairs = {}   
        singles = [] 
        
        all_files = list(src_root.rglob("*"))
        grouped = {}

        for f in all_files:
            if f.is_file():
                if f.stem not in grouped: grouped[f.stem] = {"img": None, "vid": None, "others": []}
                
                if f.suffix in img_exts:
                    grouped[f.stem]["img"] = f
                elif f.suffix in vid_exts:
                    grouped[f.stem]["vid"] = f
                else:
                    grouped[f.stem]["others"].append(f)

        valid_pairs = []
        
        for stem, data in grouped.items():
            if data["img"] and data["vid"]:
                valid_pairs.append(data)
            else:
                if data["img"]: singles.append(data["img"])
                if data["vid"]: singles.append(data["vid"])
            
            for o in data["others"]:
                singles.append(o)

        self.msg_queue.put(f"Found {len(valid_pairs)} Live Photos.")
        self.msg_queue.put(f"Found {len(singles)} regular files to copy.")

        completed = 0
        total = len(valid_pairs)

        dst_root.mkdir(parents=True, exist_ok=True)
        if valid_pairs:
            max_hint = min(8, max(1, cpu_count() - 1))
            workers = safe_worker_count(max_hint)
            avail = get_available_memory_mb()
            self.msg_queue.put(f"Available RAM: {avail} MB; Processing {len(valid_pairs)} Live Photos with {workers} worker(s)...")
            try:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = []
                    for i, pair in enumerate(valid_pairs):
                        self.msg_queue.put(f"Queued Live Photo [{i+1}/{len(valid_pairs)}]: {pair['img'].stem}")
                        futures.append(ex.submit(
                            process_live_pair,
                            pair["img"], pair["vid"], dst_root,
                            lambda m: self.msg_queue.put(m),
                            self.use_ai2.get(), self.ai2_auto_pass.get(), self
                        ))

                    for fut in as_completed(futures):
                        fut.result()
                        completed += 1
                        self.msg_queue.put(f"Completed {completed} / {total} Live Photos")

            except (MemoryError, OSError) as e:
                self.msg_queue.put(f"Low memory or resource error ({e}). Falling back to sequential processing.")
                for i, pair in enumerate(valid_pairs):
                    self.msg_queue.put(f"Sequential Live Photo [{i+1}/{len(valid_pairs)}]: {pair['img'].stem}")
                    try:
                        process_live_pair(pair['img'], pair['vid'], dst_root, lambda m: self.msg_queue.put(m), self.use_ai2.get(), self.ai2_auto_pass.get(), self)
                    except Exception as e2:
                        self.msg_queue.put(f"Sequential worker error: {e2}")

        # If manual verification mode was selected and there are flagged pairs,
        # prompt the user one-by-one on the main thread.
        # NOTE: WE DO NOT STOP THE INFERENCE WORKER YET. WE NEED IT FOR THE MANUAL APPROVED PAIRS.
        if self.use_ai2.get() and not self.ai2_auto_pass.get() and len(self.misidentified) > 0:
            self.msg_queue.put(f"{len(self.misidentified)} flagged pairs require manual verification...")
            for pair in list(self.misidentified):
                try:
                    combine = self._ask_user_sync(pair)
                    if combine:
                        self.msg_queue.put(f"User chose to combine: {pair[0].stem}")
                        # process normally but with ai2 disabled to avoid re-flagging
                        # IMPORTANT: This will use the running inference_worker for stage 2 (orientation fix)
                        try:
                            process_live_pair(pair[0], pair[1], dst_root, lambda m: self.msg_queue.put(m), False, True, self)
                        except Exception as e:
                            self.msg_queue.put(f"Error processing user-approved pair: {e}")
                    else:
                        # copy originals unchanged
                        timg = dst_root / pair[0].name
                        tv = dst_root / pair[1].name
                        if timg.exists(): timg = dst_root / f"{pair[0].stem}_{int(time.time())}{pair[0].suffix}"
                        if tv.exists(): tv = dst_root / f"{pair[1].stem}_{int(time.time())}{pair[1].suffix}"
                        copy_unchanged(pair[0], timg, lambda m: self.msg_queue.put(m))
                        copy_unchanged(pair[1], tv, lambda m: self.msg_queue.put(m))
                except Exception as e:
                    self.msg_queue.put(f"Manual verify error: {e}")
        
        # NOW we can safely stop the worker
        try:
            if inference_worker is not None:
                inference_worker.stop()
        except Exception:
            pass

        self.msg_queue.put("--- Copying other files ---")
        for i, f in enumerate(singles):
            target = dst_root / f.name
            if target.exists():
                timestamp = int(time.time())
                target = dst_root / f"{f.stem}_{timestamp}{f.suffix}"
            copy_unchanged(f, target, lambda m: self.msg_queue.put(m))

        self.msg_queue.put("Done! All tasks finished.")
        self.is_running = False
        self.progress.stop()

if __name__ == "__main__":
    freeze_support()
    root = tk.Tk()
    Apples2DroidsApp(root)
    root.mainloop()