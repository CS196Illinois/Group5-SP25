import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('galaxy_classifier.h5')

st.title("🌌 Galaxy Classifier")
st.write("Upload an image of a galaxy to classify it as Spiral or Non-Spiral.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "🌀 Spiral Galaxy" if prediction > 0.5 else "❌ Non-Spiral Galaxy"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: {label}")
    st.progress(float(confidence))
    st.write(f"**Confidence:** {confidence:.2%}")
