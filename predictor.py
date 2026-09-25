
import os
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "plant_disease_model.keras"
)


# ============================================================
# YOUR 38 MODEL CLASSES
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# CHECK WHETHER MODEL EXISTS
# ============================================================

def model_exists():
    """
    Check whether the trained model file exists.
    """

    return os.path.isfile(MODEL_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    TensorFlow is imported only when we actually need
    to load the trained model.

    Therefore the application can start without TensorFlow.
    """

    if not model_exists():
        return None

    try:

        import tensorflow as tf

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        return model

    except Exception as error:

        print(
            "Model loading error:",
            error
        )

        return None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image, model):

    """
    Prepare an uploaded image for the trained model.
    """

    image = image.convert("RGB")

    input_shape = model.input_shape

    height = input_shape[1]
    width = input_shape[2]

    image = image.resize(
        (width, height)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_disease(image, model):

    """
    Predict plant disease using the trained model.
    """

    processed_image = preprocess_image(
        image,
        model
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    ) * 100

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return (
        predicted_class,
        confidence
    )

