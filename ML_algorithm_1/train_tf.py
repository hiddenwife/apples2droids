import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
THUMBS_DIR = "training_thumbs"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32 # Change per your GPU memory (eg 16 for 4GB, 32 for 8GB+)
EPOCHS = 12
MODEL_NAME = "relative_rotation_alignment_model.keras"
VAL_SPLIT = 0.1

# ================= GPU SETUP =================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU DETECTED: {gpus}")
    except RuntimeError as e:
        print(f"GPU Config Error: {e}")
else:
    print("WARNING: No GPU detected. Using CPU.")

# ================= SAFE CUSTOM LAYERS =================

class RotateLayer(layers.Layer):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, x):
        return tf.image.rot90(x, k=self.k)

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config


class AbsLayer(layers.Layer):
    def call(self, x):
        return tf.abs(x)

# ================= DATA GENERATOR =================

class SiameseRotationGenerator(tf.keras.utils.Sequence):
    def __init__(self, files, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
        self.files = files
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        orig_batch, vers_batch, labels = [], [], []

        for path in batch_files:
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
            orig = tf.keras.preprocessing.image.img_to_array(img) / 255.0

            k = np.random.randint(0, 4)
            vers = np.rot90(orig, k=k)

            orig_batch.append(orig)
            vers_batch.append(vers)
            # Label is the rotation that SHOULD be APPLIED to `vers` to match `orig`.
            # Since `vers = rot90(orig, k)`, the required rotation is (-k) mod 4.
            labels.append((4 - k) % 4)

        return (
            {
                "original_input": np.array(orig_batch),
                "version_input": np.array(vers_batch),
            },
            np.array(labels),
        )

# ================= MODEL =================

def build_relative_rotation_model():
    input_a = layers.Input(shape=IMG_SIZE + (3,), name="original_input")
    input_b = layers.Input(shape=IMG_SIZE + (3,), name="version_input")

    def feature_extractor():
        inp = layers.Input(shape=IMG_SIZE + (3,))
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        return models.Model(inp, x)

    fe = feature_extractor()

    fa = fe(input_a)
    fb = fe(input_b)

    diffs = []
    for k in range(4):
        rotated = RotateLayer(k, name=f"rotate_{k}")(fb)
        diff = layers.Subtract()([fa, rotated])
        diff = AbsLayer()(diff)
        diff = layers.GlobalAveragePooling2D()(diff)
        diffs.append(diff)

    x = layers.Concatenate()(diffs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(4, activation="softmax")(x)

    return models.Model([input_a, input_b], output)

# ================= TRAIN =================

if __name__ == "__main__":

    all_files = [
        os.path.join(THUMBS_DIR, f)
        for f in os.listdir(THUMBS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    train_files, val_files = train_test_split(
        all_files, test_size=VAL_SPLIT, random_state=42
    )

    train_gen = SiameseRotationGenerator(train_files)
    val_gen = SiameseRotationGenerator(val_files)

    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Train steps: {len(train_gen)}")
    print(f"Validation steps: {len(val_gen)}")

    model = build_relative_rotation_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training model...")
    model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=EPOCHS,
        verbose=1,
    )

    model.save(MODEL_NAME)
    print(f"Model saved as {MODEL_NAME}")
