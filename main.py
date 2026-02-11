import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Down Syndrome Detection", layout="centered")

st.title("🧬 Down Syndrome Detection")
st.write("Upload a facial image to get prediction")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# -----------------------------
# Image uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Preprocess image
    # -----------------------------
    height, width = 64, 64
    image = image.resize((width, height))

    image_array = np.asarray(image) / 255.0
    image_array = image_array.reshape(1, width, height, 3)

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict"):
        prediction = model.predict(image_array)

        if prediction[0][0] >= 0.5:
            result = "✅ Healthy"
        else:
            result = "⚠️ Down Syndrome"

        st.subheader("Prediction:")
        st.success(result)
