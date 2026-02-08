import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

# ================= CONFIG =================
DATASET_DIR = "live_verification_dataset"
IMG_SIZE = (224, 224)
FRAMES_PER_SAMPLE = 7
BATCH_SIZE = 16
EPOCHS = 12
MODEL_NAME = "live_photo_verifier_model.keras"
# =========================================


from ML_algorithm_2.custom_layers import AbsDiff, ReduceMin, ReduceMean


# ================= DATA GENERATOR =================
class MultiFrameSetGenerator(tf.keras.utils.Sequence):
    def __init__(self, still_dir, frame_dir, stems, batch_size=16):
        self.still_dir = still_dir
        self.frame_dir = frame_dir
        self.stems = stems
        self.batch_size = batch_size

        self.frame_map = {}
        for f in os.listdir(frame_dir):
            stem = f.split("__")[0]
            self.frame_map.setdefault(stem, []).append(f)

    def __len__(self):
        return len(self.stems) // self.batch_size

    def __getitem__(self, idx):
        batch_stems = self.stems[idx*self.batch_size:(idx+1)*self.batch_size]

        stills, frames, labels = [], [], []

        for stem in batch_stems:
            still_path = self.still_dir / f"{stem}.jpg"
            if not still_path.exists():
                continue

            still = img_to_array(
                load_img(still_path, target_size=IMG_SIZE)
            ) / 255.0

            is_match = random.random() > 0.5

            if is_match:
                chosen_stem = stem
                label = 1.0
            else:
                chosen_stem = random.choice([s for s in self.stems if s != stem])
                label = 0.0

            available = self.frame_map.get(chosen_stem)
            if not available:
                continue

            frame_files = (
                random.sample(available, FRAMES_PER_SAMPLE)
                if len(available) >= FRAMES_PER_SAMPLE
                else random.choices(available, k=FRAMES_PER_SAMPLE)
            )

            frame_imgs = [
                img_to_array(
                    load_img(self.frame_dir / f, target_size=IMG_SIZE)
                ) / 255.0
                for f in frame_files
            ]

            stills.append(still)
            frames.append(np.stack(frame_imgs))
            labels.append(label)

        return (np.stack(stills), np.stack(frames)), np.array(labels)


# ================= MODEL =================
def build_model():
    base = applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    base.trainable = False

    still_in = layers.Input(shape=IMG_SIZE + (3,), name="still")
    frames_in = layers.Input(
        shape=(FRAMES_PER_SAMPLE,) + IMG_SIZE + (3,),
        name="frames"
    )

    encoder_input = layers.Input(shape=IMG_SIZE + (3,))
    x = base(encoder_input)
    x = layers.GlobalAveragePooling2D()(x)
    encoder = models.Model(encoder_input, x, name="encoder")

    still_feat = encoder(still_in)
    frame_feats = layers.TimeDistributed(encoder)(frames_in)

    still_expand = layers.RepeatVector(FRAMES_PER_SAMPLE)(still_feat)
    diffs = AbsDiff()([still_expand, frame_feats])

    min_diff = ReduceMin()(diffs)
    mean_diff = ReduceMean()(diffs)

    merged = layers.Concatenate()([min_diff, mean_diff])
    out = layers.Dense(1, activation="sigmoid")(merged)

    return models.Model([still_in, frames_in], out)


# ================= TRAIN =================
if __name__ == "__main__":
    stills = Path(DATASET_DIR) / "stills"
    frames = Path(DATASET_DIR) / "frames"

    stems = sorted({
        f.stem.split("__")[0]
        for f in frames.iterdir()
        if (stills / f"{f.stem.split('__')[0]}.jpg").exists()
    })

    random.shuffle(stems)
    split = int(len(stems) * 0.8)

    train_gen = MultiFrameSetGenerator(stills, frames, stems[:split], BATCH_SIZE)
    val_gen = MultiFrameSetGenerator(stills, frames, stems[split:], BATCH_SIZE)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    model.save(MODEL_NAME)

    print(f"Model saved to {MODEL_NAME}")
