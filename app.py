import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Function to load and prepare the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((32, 32))
    img = np.array(img)
    if img.shape[-1] == 4:  
        img = img[..., :3]  
    img = img.reshape(1, 32, 32, 3)
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
model = load_model('model1.h5')  

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
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "About the Project"])

    if page == "Home":
        # Title
        st.title("Application")

        # Main page content
        st.write("Welcome to the Leaf spot Classification App! This app uses a Convolutional Neural Network (CNN) model to classify images")
        st.write("Upload an image and the app will predict whether it has a disease")

        # List of sports categories
        health_categories = [
            
        ]

        st.write(health_categories)

    elif page == "Prediction":
        # Prediction page
        st.title("Model Prediction")
        st.write("Upload an image to predict the condition of the leaf.")

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
        This Streamlit app uses a Convolutional Neural Network (CNN) model to classify different condition categories.  
        """)
        st.write("Developed by: Team 8 (CPE32S9)")
        st.write("- Duque, Jethro")
        st.write("- Natiola, Henry Jay")

if _name_ == "_main_":
    main()
