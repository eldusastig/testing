import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

@st.cache_resource(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model1.h5')
    return model

model = load_model()

st.write("""
# Bean Leaf Lesion Identifier (Healthy, Angular Leaf Spot, Bean Rust)
""")

file = st.file_uploader("Choose a bean leaf photo", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    
    if img.ndim == 2:  # if the image is grayscale, convert to 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img = img / 255.0  # Normalize the image to the range [0, 1]
    img_reshape = img.reshape((1, 128, 128, 3))
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Healthy', 'Angular Leaf Spot', 'Bean Rust']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
