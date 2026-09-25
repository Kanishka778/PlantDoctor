import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from disease_info import DISEASE_INFO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Doctor",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "model2_efficientnet_best.keras"
CLASS_PATH = "model2_class_names.json"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(CLASS_PATH):
    st.error(f"❌ Class file not found: {CLASS_PATH}")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_plant_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD MODEL + CLASSES
# ============================================================

try:

    model = load_plant_model()
    class_names = load_class_names()

except Exception as e:

    st.error("❌ Could not load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# VERIFY MODEL
# ============================================================

if len(class_names) != 38:

    st.error(
        f"❌ Expected 38 classes but found "
        f"{len(class_names)}."
    )

    st.stop()


if model.output_shape[-1] != 38:

    st.error(
        f"❌ Model output has "
        f"{model.output_shape[-1]} classes instead of 38."
    )

    st.stop()


# ============================================================
# HELPER: MATCH DISEASE INFORMATION
# ============================================================

def get_disease_details(predicted_class):

    def clean_key(text):

        return (
            text
            .replace(" ", "")
            .replace("_", "")
            .lower()
        )

    if predicted_class in DISEASE_INFO:

        return DISEASE_INFO[predicted_class]

    target = clean_key(predicted_class)

    for key, value in DISEASE_INFO.items():

        if clean_key(key) == target:

            return value

    return None


# ============================================================
# HELPER: DISPLAY NAME
# ============================================================

def format_prediction_name(class_name):

    parts = class_name.split("___", 1)

    if len(parts) != 2:

        return class_name

    crop = parts[0].replace("_", " ")

    disease = parts[1].replace("_", " ")

    return f"{crop} — {disease}"


# ============================================================
# UI HEADER
# ============================================================

st.title("🌱 Plant Doctor")

st.write(
    "Upload a clear photograph of a plant leaf "
    "to detect a possible disease using AI."
)


# ============================================================
# PHOTO QUALITY TIPS
# ============================================================

with st.expander("📸 How to get better results"):

    st.markdown("""
    **For a more reliable prediction:**

    - 📸 Take a closer photograph
    - 🍃 Keep one leaf in the frame
    - ☀️ Use good natural lighting
    - 🚫 Avoid blurry images
    - 🔍 Show the affected portion clearly
    - 🎯 Keep the leaf centered
    - 🌑 Avoid strong shadows and glare
    - 🌿 Avoid having many leaves or plants in one image
    """)


# ============================================================
# CROP FILTER
# ============================================================

unique_crops = sorted(
    list(
        set(
            [
                c.split("___")[0]
                .replace("_", " ")
                for c in class_names
            ]
        )
    )
)


selected_crop = st.selectbox(
    "🌿 Optional: Select your target crop",
    options=["All Crops"] + unique_crops
)


# ============================================================
# IMAGE INPUT
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload plant leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# RESET WHEN NEW IMAGE IS UPLOADED
# ============================================================

if "last_uploaded_file" not in st.session_state:

    st.session_state["last_uploaded_file"] = None


if uploaded_file != st.session_state["last_uploaded_file"]:

    st.session_state["last_uploaded_file"] = uploaded_file

    st.session_state["analyzed"] = False

    for key in [
        "predicted_class",
        "confidence",
        "top_predictions"
    ]:

        if key in st.session_state:

            del st.session_state[key]


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Uploaded leaf",
        width="stretch"
    )


    st.success(
        "✅ Image uploaded successfully!"
    )


    # ========================================================
    # DETECT BUTTON
    # ========================================================

    if st.button(
        "🔍 Detect Disease",
        type="primary",
        width="stretch"
    ):

        try:

            with st.spinner(
                "🌿 Analyzing your plant..."
            ):

                # --------------------------------------------
                # PREPROCESS IMAGE
                # --------------------------------------------

                resized = image.resize(
                    (224, 224)
                )

                image_array = np.asarray(
                    resized,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )


                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]


                # --------------------------------------------
                # CROP FILTER
                # --------------------------------------------

                filtered_predictions = predictions.copy()


                if selected_crop != "All Crops":

                    cleaned_target = (
                        selected_crop
                        .lower()
                        .replace(" ", "")
                        .replace("_", "")
                    )


                    matching_indices = []


                    for idx, name in enumerate(
                        class_names
                    ):

                        cleaned_name = (
                            name
                            .lower()
                            .replace("_", "")
                        )


                        if cleaned_name.startswith(
                            cleaned_target
                        ):

                            matching_indices.append(
                                idx
                            )


                    if matching_indices:

                        mask = np.full(
                            len(predictions),
                            -1.0
                        )


                        for idx in matching_indices:

                            mask[idx] = predictions[idx]


                        filtered_predictions = mask


                # --------------------------------------------
                # TOP 3
                # --------------------------------------------

                top_indices = np.argsort(
                    filtered_predictions
                )[::-1][:3]


                top_predictions = []


                for idx in top_indices:

                    if filtered_predictions[idx] >= 0:

                        top_predictions.append(
                            (
                                class_names[idx],
                                float(
                                    predictions[idx]
                                ) * 100
                            )
                        )


                if not top_predictions:

                    st.error(
                        "Unable to find a suitable "
                        "prediction for the selected crop."
                    )

                    st.stop()


                predicted_class = (
                    top_predictions[0][0]
                )

                confidence = (
                    top_predictions[0][1]
                )


                # --------------------------------------------
                # SAVE RESULTS
                # --------------------------------------------

                st.session_state[
                    "predicted_class"
                ] = predicted_class


                st.session_state[
                    "confidence"
                ] = confidence


                st.session_state[
                    "top_predictions"
                ] = top_predictions


                st.session_state[
                    "analyzed"
                ] = True


        except Exception as error:

            st.error(
                f"❌ Unable to analyze the image: {error}"
            )

            st.session_state[
                "analyzed"
            ] = False


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if st.session_state.get(
        "analyzed",
        False
    ):

        predicted_class = (
            st.session_state[
                "predicted_class"
            ]
        )


        confidence = (
            st.session_state[
                "confidence"
            ]
        )


        top_predictions = (
            st.session_state[
                "top_predictions"
            ]
        )


        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.divider()

        st.header("🌿 Plant Doctor Result")


        # ====================================================
        # CONFIDENCE LEVEL
        # ====================================================

        if confidence >= 80:

            confidence_label = "🟢 High confidence"

        elif confidence >= 60:

            confidence_label = "🟡 Moderate confidence"

        else:

            confidence_label = "🔴 Low confidence"


        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )

        st.write(
            f"**{confidence_label}**"
        )


        # ====================================================
        # MAIN PREDICTION
        # ====================================================

        st.subheader(
            "🔬 Most likely result"
        )


        st.success(
            format_prediction_name(
                predicted_class
            )
        )


        # ====================================================
        # TOP 3
        # ====================================================

        st.subheader(
            "📊 Top 3 AI predictions"
        )


        for position, (
            prediction_class,
            prediction_confidence
        ) in enumerate(
            top_predictions,
            start=1
        ):

            st.write(
                f"**{position}. "
                f"{format_prediction_name(prediction_class)}**"
                f" — "
                f"{prediction_confidence:.2f}%"
            )


        # ====================================================
        # LOW CONFIDENCE WARNING
        # ====================================================

        if confidence < 60:

            st.warning(
                f"""
                ⚠️ **The AI is not sufficiently confident
                in this prediction ({confidence:.2f}%).**

                Please take another photograph using:

                • Good natural lighting  
                • One leaf in the frame  
                • A closer view  
                • A sharp, non-blurry image  
                • The affected portion clearly visible
                """
            )


        elif confidence < 80:

            st.info(
                f"""
                ℹ️ The AI has moderate confidence
                ({confidence:.2f}%).

                For a more reliable result, try another
                clear photograph following the photo tips above.
                """
            )


        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

        info = get_disease_details(
            predicted_class
        )


        if info:

            st.divider()

            st.header(
                "📖 About the prediction"
            )


            st.write(
                f"**Crop:** {info['crop']}"
            )


            st.write(
                f"**Disease:** {info['disease']}"
            )


            st.subheader(
                "What does this mean?"
            )


            st.write(
                info["about"]
            )


            st.subheader(
                "💡 What should you do?"
            )


            st.write(
                info["action"]
            )


            st.subheader(
                "🛡️ Prevention"
            )


            st.write(
                info["prevention"]
            )


        else:

            st.info(
                "Detailed information for this disease "
                "has not been added yet."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 Plant Doctor uses an AI image-classification "
    "model to provide a possible plant-disease prediction. "
    "Results should be treated as an AI-assisted indication, "
    "not a guaranteed agricultural diagnosis."
)