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
model = load_model('model8.hdf5')  

# Function to load labels from a text file
def load_labels(filename):
    with open(filename, 'r') as file:
        labels = file.readlines()
    labels = [label.strip() for label in labels]
    return labels

# Streamlit UI
def main():
    # Sidebar
    st.sidebar.title("TEAM 8 Model Deployment in the Cloud")
    page = st.sidebar.radio("Go to", ["Model", "About the Project"])

    if page == "Model":
        # Title
        st.title("Malaria Cell Classifier")

        # Main page content
        st.write("Welcome to the Malaria Detection app! This app uses a Convolutional Neural Network (CNN) model to classify if a cell is infected with Malaria or not")
        st.write("Upload an image and the app will predict whether an cell is infected or not")
        st.write("Uploaded Image should only contain ONE cell")
        malaria_banner = "https://raw.githubusercontent.com/eldusastig/testing/blob/main/malaria.png"  # Replace this URL with the URL of your image
        st.image( malaria_banner , caption='Like this ', use_column_width=True)
        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            st.image(test_image, width=300, caption='Uploaded Image')
            if st.button("Predict"):
                    st.write("Predicting...")
                    labels = load_labels("labels.txt")
                    predicted_health = predict(test_image, model, labels)
                    st.success(f"Predicted Condition Category: {predicted_health}")

    
    elif page == "About the Project":
        # About the project
        st.title("About the Project")
        st.write("""
        This Streamlit app uses a Convolutional Neural Network (CNN) model that classifies cell for Malaria Detection.  
        """)
        st.write("Developed by: Team 8 (CPE32S9)")
        st.write("- Duque, Jethro")
        st.write("- Natiola, Henry Jay")

if __name__ == "__main__":
    main()
