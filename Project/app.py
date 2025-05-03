import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random

#page configuration
st.set_page_config(page_title="Spiral.Ai", layout="wide")

#CSS 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif !important;
        background-color: #0a0a23;
        color: white;
    }

    h1, h2, h3, h4, .css-10trblm, .css-1v0mbdj, .css-1d391kg {
        font-family: 'Orbitron', sans-serif !important;
        color: #0a0a23 !important;
    }

    .stSidebar, .css-1d391kg {
        background-color: #f0f2f6;
    }

    .stButton button {
        background-color: #008ae6;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
        transition: background-color 0.3s ease;
    }

    .stButton button:hover {
        background-color: #004d80;
    }

    .stProgress > div > div > div > div {
        background-color: #008ae6;
    }

    </style>
    """,
    unsafe_allow_html=True
)

#loading model
model = tf.keras.models.load_model('galaxy_classifier.h5')

#side bar
st.sidebar.title("🪐 Spiral.Ai")
st.sidebar.markdown("""


Have you ever looked at a mesmerizing galaxy image and wondered whether it's a spiral or not.
Be uncertain no more!


Inspired by our curiosity for astronomy and how machines see the universe.


🧠 CNN model trained on space imagery  
🛠️ Built with TensorFlow + Streamlit  

Created by Ahmad Daye & Navika Tewari

""")

#header
st.title("✨ Spiral.Ai")
st.markdown("""
Welcome to **Spiral.Ai** — our AI driven galaxy classifier.  
Upload a galaxy image and to discover whether it’s a **Spiral** galaxy 🌀 or **Non-Spiral** ❌.

---
""")

#image upload section
st.subheader("📤 Upload a Galaxy Image")
uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🔭 Uploaded Galaxy", use_container_width=True)

    #preprocessing
    img = image.convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "🌀 Spiral Galaxy" if prediction > 0.5 else "❌ Non-Spiral Galaxy"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🌠 Prediction: {label}")
    st.progress(float(confidence))
    st.write(f"**Confidence Score:** {confidence:.2%}")


#galaxy context
st.markdown("---")
st.subheader("🌠 What Are Spiral and Non-Spiral Galaxies?")

st.markdown("""
Galaxies are made up of stars, planets, and vast clouds of gas and dust, all bound together by gravity. 
They come in many shapes, but two major types are:

- **🌀 Spiral Galaxies**: These have a bright center with a tight concentration of stars and with arms that spiral outward.
  They look like giant pinwheels! But from Earth we only see a side view so the arms cannot be seen.
  They’re rich in gas, dust, and have younger brighter stars
  — the Milky Way is one!

- **🔵 Non-Spiral Galaxies**: These are any galaxies that are not spiral, including **elliptical** and **irregular** galaxies.  
  They lack distinct arms and have chaotic shapes. They are often older, with less star formation activity
  or formed by collisions and interactions with other galaxies losing their distint original shaps. 

""")

st.markdown("---")

st.markdown("""
## 🧭 How It Works

During training, our model was shown a large set of galaxy images labeled as **spiral** or **non-spiral**.  
It learned to detect morphologcical features like curves, arms, symmetry, and brightness that help differentiate between the two.

When you upload a new image, Spiral.Ai processes it and feeds it through our trained model to generate a prediction,
telling you how likely it is to be a spiral galaxy based on what it has learned.


---
""")

#fun fact section
facts = [
    "The Milky Way rotates once every 240 million years.",
    "Elliptical galaxies are typically older and redder.",
    "A future predicted collision is between the Milky Way and Andromeda Galaxy.",
    "Dark matter trendrils explains why spiral galaxies rotate the way they do.",
    "The name Milky Way comes from a Greek Myth about Hera nursing Heracles.",
    "There are 100 billion galaxies in the observable universe.",
    "Most galaxies have a black hole at the center.",
]

col1, col2 = st.columns([6, 1])

with col1:
    st.info(f"🌌 Fun Fact: {random.choice(facts)}")

with col2:
    st.write("")
    if st.button("🔁 New Fact"):
        st.rerun()
