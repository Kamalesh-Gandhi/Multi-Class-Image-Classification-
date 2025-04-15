import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import os
import base64

def get_class_names_from_dir(path='data/test'):
    classes = sorted(entry.name for entry in os.scandir(path) if entry.is_dir())
    return classes

# ✅ Encode local video
def set_fullscreen_image_bg(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        /* FULL BACKGROUND IMAGE */
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* GLASSMORPHIC CONTAINER */
        .glass-card {{
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2rem 3rem;
            margin: 2rem auto;
            max-width: 800px;
            color: #000;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }}

        .stMarkdown h1, .stMarkdown h3, .stMarkdown p {{
            color: #000 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
bg_path = os.path.join("Background Img", "backgroundimg.png")
set_fullscreen_image_bg(bg_path)


# ✅ App title
st.title("🐟 Fish Image Classifier")

# ✅ Load model (change to your best model path)
MODEL_PATH = 'models/best_fish_classifier_model.h5' 
model = load_model(MODEL_PATH)

# ✅ Define image size (match training)
IMG_SIZE = (224, 224)

# ✅ Class labels (ensure order matches model training)
class_names = get_class_names_from_dir('data/test')


# ✅ Image upload
uploaded_file = st.file_uploader("**Upload a fish image**", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Load and preprocess
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize(IMG_SIZE)
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence_scores = prediction[0]

    # ✅ Show results
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.markdown(f"### 🔍 Predicted Class: **{predicted_class}**")
    
    # ✅ Show confidence bar chart
    st.subheader("📊 Prediction Confidence:")
    chart_data = {class_names[i]: float(confidence_scores[i]) for i in range(len(class_names))}
    st.bar_chart(chart_data)


# ✅ Close glass container
st.markdown('</div>', unsafe_allow_html=True)
