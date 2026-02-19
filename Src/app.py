import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model("Model/cat_dog_model.keras")

# Title
st.title("🐶🐱 Cat vs Dog Classifier")

st.write("Upload an image and the model will predict Cat or Dog")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.success(f"Prediction: DOG 🐶 ({confidence:.2f})")
    else:
        st.success(f"Prediction: CAT 🐱 ({1-confidence:.2f})")