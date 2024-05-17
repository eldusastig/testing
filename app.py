from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model3.h5')
    return model

model = load_model()

st.write("# Rose Leaf Disease Classification")

file = st.file_uploader("Choose a rose leaf photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (32, 32)
    
    # Resize the image using PIL's resize function
    image = image_data.resize(size)
    img = np.asarray(image)
    
    # Convert the image to grayscale if necessary
    if img.ndim == 3 and img.shape[2] == 3:
        img = np.dot(img, [0.2989, 0.5870, 0.1140])  # Convert to grayscale using luminosity method

    # Reshape the image to add a channel dimension
    img_reshape = img.reshape((1,) + img.shape + (1,))

    # Make predictions using the Keras model
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Healthy', 'Rose Rust',  'Rose Slug Sawfly']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
