import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Function to load and prepare the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((50, 50))
    img = np.array(img)
    if img.shape[-1] == 4:  
        img = img[..., :3]  
    img = img.reshape(1, 50, 50, 3)
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to make predictions
def predict(image, model, labels):
    img = load_image(image)
    result = model.predict(img)
    predicted_class = np.argmax(result, axis=1)
    return labels[predicted_class[0]]

# Load the model
model = load_model('new_model.h5')  

# Function to load labels from a text file
def load_labels(filename):
    with open(filename, 'r') as file:
        labels = file.readlines()
    labels = [label.strip() for label in labels]
    return labels

# Streamlit UI
def main():
    st.sidebar.title("TEAM 6 Model Deployment in the Cloud")
    st.title("Rice Classifier")

    # Main page content
    st.write("Welcome to the Rice Classification App")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, width=300, caption='Uploaded Image')
        if st.button("Predict"):
            st.write("Predicting...")
            labels = load_labels("labels.txt")
            predicted_health = predict(test_image, model, labels)
            st.success(f"Predicted Rice Category: {predicted_health}")

if __name__ == "__main__":
    main()
