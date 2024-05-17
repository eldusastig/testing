import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import os

# Function to load and prepare the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    img = np.array(img)
    img = img.reshape(1, 128, 128, 3)
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to make predictions
def import_and_predict(image_data, model):
    size = (128, 128)
    
    # Resize the image to the expected input shape of the model
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    # Convert the image to grayscale if necessary
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Reshape the image to add a channel dimension
    img_reshape = img.reshape((1,) + img.shape + (1,))

    # Make predictions using the Keras model
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    string="OUTPUT : "+ class_names[np.argmax(prediction)]
    st.success(string)


# Load the model
model = load_model('model3.h5')

# Define the class labels
class_labels = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']

# Streamlit UI
def main():
    # Sidebar
    st.sidebar.title("TEAM 8 Model Deployment in the Cloud")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "About the Project"])

    if page == "Home":
        # Title
        st.title("Application")

        # Main page content
        st.write("Welcome to the Rose Leaf Classification App! This app uses a Convolutional Neural Network (CNN) model to classify images")
        st.write("Upload an rose image and the app would classify if it's Healthy, Rose Rust or have Rose Slug Sawfly damage")

        # List of health categories
        health_categories = [
            "Healthy",
            "Rose Rust",
            "Rose Slug Sawfly damage"
        ]

        st.write("Health Categories:", health_categories)

    elif page == "Prediction":
        # Prediction page
        st.title("Model Prediction")
        st.write("Upload an image to predict the condition of the leaf.")

        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            st.image(test_image, width=300, caption='Uploaded Image')
            if st.button("Predict"):
                st.write("Predicting...")
                predicted_health, confidence = predict_image(test_image, model)
                st.write(f"Predicted Condition Category: {predicted_health}")
                st.write(f"Confidence: {confidence:.2f}")

    elif page == "About the Project":
        # About the project
        st.title("About the Project")
        st.write("""
        This Streamlit app uses a Convolutional Neural Network (CNN) model to classify different condition categories.  
        """)
        st.write("Developed by: Team 8 (CPE32S9)")
        st.write("- Duque, Jethro")
        st.write("- Natiola, Henry Jay")

if __name__ == "__main__":
    main()
