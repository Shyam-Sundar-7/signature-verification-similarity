import streamlit as st
import numpy as np
import PIL
from PIL import Image
from io import BytesIO
import requests
from inference import Inference_prediction

# Define the Streamlit app
def main():
    # Set up the Streamlit app
    st.set_page_config(page_title="Image Similarity App", page_icon=":guardsman:", layout="wide")
    st.title("Image Similarity App")

    # Get the two images from the user
    col1, col2 = st.columns(2)
    img1 = col1.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    img2 = col2.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

    # Display the uploaded images
    if img1:
        img1 = Image.open(img1)
        col1.image(img1, caption="Image 1", use_column_width=True)
    if img2:
        img2 = Image.open(img2)
        col2.image(img2, caption="Image 2", use_column_width=True)

    # Compute the features and similarity score
    if img1 is not None and img2 is not None:
        if st.button("Compute Similarity"):
            # Extract features from the images
            inference = Inference_prediction("best_model.ckpt")
        
            x,d=inference.predict(img1,img2)

            # Display the similarity score
            st.markdown(f"The similarity score between the two images is **{d:.2f}**.")
            g="Forged" if x>50 else "Original"
            st.markdown(f"The Testing Signature is {g}")

# Run the Streamlit app
if __name__ == "__main__":
    main()