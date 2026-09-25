
# ============================================================
# PLANT DOCTOR - COMPLETE APPLICATION
# Model 2: EfficientNetB0 - 38 Plant Disease Classes
# ============================================================

import os
import json
import numpy as np
from PIL import Image

import streamlit as st
import tensorflow as tf

# Gemini AI
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
# PATH DETECTION
# ============================================================

# Your files may be in the PlantDoctor root folder
# or inside a Models folder.
#
# The code checks both automatically.

MODEL_LOCATIONS = [
    os.path.join("model2_efficientnet_best.keras"),
    os.path.join("models", "model2_efficientnet_best.keras"),
    os.path.join("Models", "model2_efficientnet_best.keras"),
]

CLASS_LOCATIONS = [
    os.path.join("model2_class_names.json"),
    os.path.join("models", "model2_class_names.json"),
    os.path.join("Models", "model2_class_names.json"),
]


def find_existing_file(locations):

    for path in locations:

        if os.path.exists(path):
            return path

    return None


MODEL_PATH = find_existing_file(MODEL_LOCATIONS)
CLASS_PATH = find_existing_file(CLASS_LOCATIONS)


# ============================================================
# CHECK MODEL
# ============================================================

if MODEL_PATH is None:

    st.error(
        "❌ Model 2 could not be found.\n\n"
        "Expected file:\n"
        "`model2_efficientnet_best.keras`"
    )

    st.stop()


# ============================================================
# CHECK CLASS FILE
# ============================================================

if CLASS_PATH is None:

    st.error(
        "❌ Class file could not be found.\n\n"
        "Expected file:\n"
        "`model2_class_names.json`"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_plant_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


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
# LOAD EVERYTHING
# ============================================================

try:

    model = load_plant_model()

    class_names = load_class_names()

except Exception as error:

    st.error(
        "❌ Could not load PlantDoctor."
    )

    st.exception(error)

    st.stop()


# ============================================================
# VERIFY 38 CLASSES
# ============================================================

if len(class_names) != 38:

    st.error(
        f"❌ Expected 38 classes but found "
        f"{len(class_names)}."
    )

    st.stop()


# ============================================================
# VERIFY MODEL OUTPUT
# ============================================================

try:

    output_classes = model.output_shape[-1]

    if output_classes != 38:

        st.error(
            f"❌ Model output contains "
            f"{output_classes} classes instead of 38."
        )

        st.stop()

except Exception as error:

    st.error(
        "❌ Could not verify model output."
    )

    st.exception(error)

    st.stop()


# ============================================================
# DISEASE INFORMATION HELPER
# ============================================================

def get_disease_details(predicted_class):

    def clean_key(text):

        return (
            text
            .replace(" ", "")
            .replace("_", "")
            .lower()
        )

    # Exact match
    if predicted_class in DISEASE_INFO:

        return DISEASE_INFO[predicted_class]

    # Cleaned match
    target = clean_key(
        predicted_class
    )

    for key, value in DISEASE_INFO.items():

        if clean_key(key) == target:

            return value

    return None


# ============================================================
# GEMINI AI FUNCTION
# ============================================================

def ask_plantdoctor_ai(
    predicted_class,
    confidence,
    info,
    user_question
):

    # --------------------------------------------------------
    # CHECK GEMINI PACKAGE
    # --------------------------------------------------------

    if not GEMINI_AVAILABLE:

        return (
            "⚠️ Gemini AI package is not installed.\n\n"
            "Run this in the VS Code terminal:\n\n"
            "`.venv\\Scripts\\python.exe -m pip "
            "install -U google-genai`"
        )


    # --------------------------------------------------------
    # GET API KEY
    # --------------------------------------------------------

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            "⚠️ Gemini API key was not found.\n\n"
            "Please set the Windows environment variable "
            "`GEMINI_API_KEY` and restart VS Code/Streamlit."
        )


    try:

        # ----------------------------------------------------
        # CREATE GEMINI CLIENT
        # ----------------------------------------------------

        client = genai.Client(
            api_key=api_key
        )


        # ----------------------------------------------------
        # GET VERIFIED INFORMATION
        # ----------------------------------------------------

        crop = info.get(
            "crop",
            "Unknown"
        )

        disease = info.get(
            "disease",
            predicted_class
        )

        about = info.get(
            "about",
            "Information not available."
        )

        action = info.get(
            "action",
            "Information not available."
        )

        prevention = info.get(
            "prevention",
            "Information not available."
        )


        # ----------------------------------------------------
        # AI PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are PlantDoctor AI, an agricultural assistant.

PlantDoctor is a plant disease detection application
using a deep-learning image classification model.

The model has made the following prediction:

Crop:
{crop}

Predicted condition:
{disease}

Model confidence:
{confidence:.2f}%

Existing verified PlantDoctor information:

ABOUT:
{about}

WHAT TO DO:
{action}

PREVENTION:
{prevention}

User question:
{user_question}

Instructions:

1. Answer the user's question clearly and simply.
2. Give practical information useful to a farmer or student.
3. Do not claim that the model diagnosis is 100% certain.
4. Do not invent pesticide names or dosage amounts.
5. If chemical treatment is discussed, advise the user to
   consult a qualified agricultural expert for the correct
   product and dosage.
6. Explain difficult agricultural terms simply.
7. Keep the answer concise but useful.
8. Base the response primarily on the detected crop,
   condition and verified information supplied above.
"""

        # ----------------------------------------------------
        # SEND REQUEST
        # ----------------------------------------------------

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        # ----------------------------------------------------
        # RETURN RESPONSE
        # ----------------------------------------------------

        if response is not None:

            if getattr(
                response,
                "text",
                None
            ):

                return response.text

        return (
            "⚠️ Gemini returned an empty response. "
            "Please try again."
        )


    except Exception as error:

        return (
            "⚠️ PlantDoctor AI could not generate "
            "an answer.\n\n"
            f"Error: {error}"
        )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🌱 Plant Doctor"
)

st.write(
    "Upload a clear photograph of a plant leaf "
    "to detect a possible disease."
)


# ============================================================
# HOW TO GET BEST RESULTS
# ============================================================

st.info(
    """
### 📷 Tips for better results

• Take a closer photograph  
• Keep one leaf in the frame  
• Use good natural lighting  
• Avoid blurry images  
• Show the affected portion clearly  
"""
)


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
    options=[
        "All Crops"
    ] + unique_crops
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_uploaded_file" not in st.session_state:

    st.session_state[
        "last_uploaded_file"
    ] = None


if "analyzed" not in st.session_state:

    st.session_state[
        "analyzed"
    ] = False


if uploaded_file != st.session_state[
    "last_uploaded_file"
]:

    st.session_state[
        "last_uploaded_file"
    ] = uploaded_file

    st.session_state[
        "analyzed"
    ] = False

    st.session_state.pop(
        "predicted_class",
        None
    )

    st.session_state.pop(
        "confidence",
        None
    )

    st.session_state.pop(
        "predictions",
        None
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf",
            use_container_width=True
        )

        st.success(
            "✅ Image uploaded successfully!"
        )


    except Exception as error:

        st.error(
            "❌ Could not open the image."
        )

        st.exception(error)

        st.stop()


    # ========================================================
    # DETECT DISEASE BUTTON
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
                # RESIZE
                # --------------------------------------------

                resized = image.resize(
                    (224, 224)
                )


                # --------------------------------------------
                # CONVERT TO ARRAY
                # --------------------------------------------

                image_array = np.asarray(
                    resized,
                    dtype=np.float32
                )


                # --------------------------------------------
                # ADD BATCH DIMENSION
                # --------------------------------------------

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

                if selected_crop != "All Crops":

                    cleaned_target = (
                        selected_crop
                        .lower()
                        .replace(" ", "")
                        .replace("_", "")
                    )

                    filtered_predictions = []

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

                            filtered_predictions.append(
                                predictions[idx]
                            )

                        else:

                            filtered_predictions.append(
                                -1.0
                            )


                    if max(
                        filtered_predictions
                    ) == -1.0:

                        predicted_index = int(
                            np.argmax(
                                predictions
                            )
                        )

                    else:

                        predicted_index = int(
                            np.argmax(
                                filtered_predictions
                            )
                        )

                else:

                    predicted_index = int(
                        np.argmax(
                            predictions
                        )
                    )


                # --------------------------------------------
                # SAVE RESULTS
                # --------------------------------------------

                predicted_class = (
                    class_names[
                        predicted_index
                    ]
                )

                confidence = float(
                    predictions[
                        predicted_index
                    ]
                ) * 100


                st.session_state[
                    "predicted_class"
                ] = predicted_class

                st.session_state[
                    "confidence"
                ] = confidence

                st.session_state[
                    "predictions"
                ] = predictions

                st.session_state[
                    "analyzed"
                ] = True


        except Exception as error:

            st.error(
                f"❌ Unable to analyze the image: "
                f"{error}"
            )

            st.exception(error)

            st.session_state[
                "analyzed"
            ] = False


    # ========================================================
    # DISPLAY RESULT
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


        # ----------------------------------------------------
        # GET DISEASE INFORMATION
        # ----------------------------------------------------

        info = get_disease_details(
            predicted_class
        )


        # ====================================================
        # LOW CONFIDENCE
        # ====================================================

        if confidence < 75.0:

            st.warning(
                f"""
⚠️ **Low-confidence prediction**

The model predicted:

**{predicted_class}**

Confidence:

**{confidence:.2f}%**

Please upload a clearer photograph with:

• One leaf clearly visible  
• Good natural lighting  
• Minimal background  
• The affected area clearly visible  
• No blur
"""
            )


            # Still show the prediction
            st.subheader(
                "🔎 Model Prediction"
            )

            st.write(
                f"**Prediction:** "
                f"{predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            # ------------------------------------------------
            # AI NOT SHOWN FOR LOW CONFIDENCE
            # ------------------------------------------------

            st.info(
                "🤖 AI guidance is available after "
                "a sufficiently confident prediction."
            )


        # ====================================================
        # GOOD CONFIDENCE
        # ====================================================

        else:

            # ------------------------------------------------
            # DISEASE INFORMATION AVAILABLE
            # ------------------------------------------------

            if info:

                st.divider()

                st.header(
                    "🌿 Diagnosis"
                )


                # --------------------------------------------
                # CROP
                # --------------------------------------------

                st.write(
                    f"**🌱 Crop:** "
                    f"{info.get('crop', 'Unknown')}"
                )


                # --------------------------------------------
                # DISEASE
                # --------------------------------------------

                st.write(
                    f"**🦠 Disease:** "
                    f"{info.get('disease', predicted_class)}"
                )


                # --------------------------------------------
                # CONFIDENCE
                # --------------------------------------------

                st.metric(
                    "🎯 Model Confidence",
                    f"{confidence:.2f}%"
                )


                # --------------------------------------------
                # ABOUT
                # --------------------------------------------

                st.subheader(
                    "📖 What does this mean?"
                )

                st.write(
                    info.get(
                        "about",
                        "Information not available."
                    )
                )


                # --------------------------------------------
                # ACTION
                # --------------------------------------------

                st.subheader(
                    "💡 What should you do?"
                )

                st.write(
                    info.get(
                        "action",
                        "Information not available."
                    )
                )


                # --------------------------------------------
                # PREVENTION
                # --------------------------------------------

                st.subheader(
                    "🛡️ Prevention"
                )

                st.write(
                    info.get(
                        "prevention",
                        "Information not available."
                    )
                )


                # =================================================
                # AI ASSISTANT
                # =================================================

                st.divider()

                st.header(
                    "🤖 PlantDoctor AI Assistant"
                )

                st.write(
                    "Ask questions about the detected "
                    "plant condition."
                )


                question = st.text_input(
                    "💬 Ask PlantDoctor AI",
                    placeholder=(
                        "Example: How can I prevent "
                        "this disease?"
                    )
                )


                if st.button(
                    "🤖 Ask AI",
                    type="secondary",
                    width="stretch"
                ):

                    if not question.strip():

                        st.warning(
                            "⚠️ Please enter a question first."
                        )

                    else:

                        with st.spinner(
                            "🤖 PlantDoctor AI is thinking..."
                        ):

                            answer = ask_plantdoctor_ai(
                                predicted_class,
                                confidence,
                                info,
                                question
                            )


                        st.markdown(
                            "### 🤖 PlantDoctor AI"
                        )

                        st.write(
                            answer
                        )


            # ------------------------------------------------
            # INFORMATION NOT AVAILABLE
            # ------------------------------------------------

            else:

                st.header(
                    "🌿 Diagnosis"
                )

                st.write(
                    f"**Prediction:** "
                    f"{predicted_class}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.info(
                    "Detailed information for this "
                    "prediction is not available, so "
                    "the AI assistant cannot provide "
                    "verified guidance."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 Plant Doctor | AI-assisted plant disease detection"
)

st.caption(
    "⚠️ This application provides an AI-based "
    "possible diagnosis and should not replace "
    "professional agricultural advice."
)
