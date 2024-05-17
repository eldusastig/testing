import streamlit as st
import tensorflow as tf
@st.cache_resource(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model1.h5')
  return model
model=load_model()
st.write("""
# Bean Leaf Lesion Identifier(Healthy , Angular Leaf Spot, Bean Ruse)"""
)
file=st.file_uploader("Choose an bean leaf photo",type=["jpg","png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (128, 128)

    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_reshape = img.reshape((1,) + img.shape + (1,))

    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Healthy', 'Angular Leaf Spot', 'Bean Rust']
    string="OUTPUT : "+ class_names[np.argmax(prediction)]
    st.success(string)
