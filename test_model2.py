import os
import json
import tensorflow as tf

MODEL_PATH = "model2_efficientnet_best.keras"
CLASS_PATH = "model2_class_names.json"

print("======================================")
print("PLANT DOCTOR - MODEL 2 TEST")
print("======================================")

print("\nChecking files...")

print("Model exists:", os.path.exists(MODEL_PATH))
print("Class file exists:", os.path.exists(CLASS_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        f"Class file not found: {CLASS_PATH}"
    )

print("\nLoading Model 2...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(
    CLASS_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print("\n======================================")
print("MODEL LOADED SUCCESSFULLY")
print("======================================")

print("Model output:", model.output_shape)
print("Number of classes:", len(class_names))

print("\nFirst 5 classes:")

for i, name in enumerate(class_names[:5]):
    print(i, name)

print("\nLast 5 classes:")

for i, name in enumerate(
    class_names[-5:],
    start=len(class_names) - 5
):
    print(i, name)

print("\n======================================")

if model.output_shape[-1] == 38 and len(class_names) == 38:
    print("✅ MODEL 2 READY FOR APPLICATION")
else:
    print("❌ CLASS/MODEL MISMATCH")