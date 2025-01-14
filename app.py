import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Dictionary for class mapping
dic = {0: 'Normal', 1: "Infected"}

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter('artifacts/converted_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Streamlit App
st.title("X-ray Infection Predictor")
st.write("Upload your chest X-ray to check if it is Normal or Infected.")

# File uploader
uploaded_file = st.file_uploader("Upload your X-ray image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        # Button for prediction
        if st.button("Predict"):
            # Convert to grayscale
            if len(np.array(image).shape) != 2:
                st.warning("Converting image to grayscale.")
                image = ImageOps.grayscale(image)
            
            # Resize image
            image = image.resize((224, 224))
            # Normalize image
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            # Set input tensor
            interpreter.set_tensor(input_details['index'], image_array)
            
            # Run inference
            interpreter.invoke()
            
            # Get prediction
            prediction = interpreter.get_tensor(output_details['index'])
            predicted_class = np.argmax(prediction, axis=-1)[0]  # Assuming single prediction
            
            st.success(f"The prediction is: {dic[predicted_class]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
