import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
DATA_DIR = "DATASET PATH"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60
SEED = 42

MODEL_PATH = "banana_model.keras"
TFLITE_PATH = "banana_model.tflite"
CLASS_NAMES_PATH = "class_names.json"

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------- Load datasets ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    seed=SEED,
    shuffle=True,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
assert NUM_CLASSES == 9, f"Expected 9 classes, found {NUM_CLASSES}: {class_names}"
print("Classes:", class_names)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ---------------- Class weights (handles uneven folders) ----------------
counts = np.array([len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in class_names])
total = counts.sum()
class_weight = {i: total / (NUM_CLASSES * c) for i, c in enumerate(counts)}
print("Training images per class:", dict(zip(class_names, counts.tolist())))

# ---------------- Augmentation (color-safe) ----------------
# These layers only run during training; they are skipped at inference.
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="augmentation",
)

# ---------------- Model (from scratch) ----------------
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def build_model(img_size, num_classes):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    x = conv_block(x, 32)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, 64)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, 128)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, 128)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, 256)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="banana_cnn")

model = build_model(IMG_SIZE, NUM_CLASSES)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    ),
]

# ---------------- Train ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
)

# ---------------- Plot training curves ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ---------------- Final test evaluation (run once) ----------------
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
y_prob = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# ---------------- Confusion matrix ----------------
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Test Set)")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ---------------- Per-class precision, recall, F1 ----------------
print("\nPer-class metrics:")
print(f"{'Class':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}")
f1_scores = []
for i, name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f1_scores.append(f1)
    print(f"{name:<20}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}")
print(f"\nMacro F1: {np.mean(f1_scores):.4f}")

# ---------------- Variety and ripeness accuracy (derived from 9-class output) ----------------
def split_label(name):
    variety, ripeness = name.rsplit("_", 1)
    return variety, ripeness

true_variety = [split_label(class_names[i])[0] for i in y_true]
pred_variety = [split_label(class_names[i])[0] for i in y_pred]
true_ripe = [split_label(class_names[i])[1] for i in y_true]
pred_ripe = [split_label(class_names[i])[1] for i in y_pred]

variety_acc = np.mean([t == p for t, p in zip(true_variety, pred_variety)])
ripeness_acc = np.mean([t == p for t, p in zip(true_ripe, pred_ripe)])
print(f"Variety accuracy:  {variety_acc:.4f}")
print(f"Ripeness accuracy: {ripeness_acc:.4f}")

# ---------------- Save model and export for mobile ----------------
model.save(MODEL_PATH)
print(f"\nSaved Keras model to {MODEL_PATH}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model to {TFLITE_PATH}")