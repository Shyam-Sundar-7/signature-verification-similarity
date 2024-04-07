import streamlit as st
from PIL import Image
from code.inference1 import Inference_prediction1
from code.inference2 import Inference_prediction2

# Define the Streamlit app
def main():
    # Set up the Streamlit app
    st.set_page_config(page_title="Image Similarity App", page_icon=":guardsman:", layout="wide")
    st.title("Image Similarity App")

    # Get the two images from the user
    col1, col2 = st.columns(2)
    img1 = col1.file_uploader("Upload Testing Signature", type=["jpg", "jpeg", "png"])
    img2 = col2.file_uploader("Upload Original Sinature", type=["jpg", "jpeg", "png"])

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
            inference1 = Inference_prediction1("saved_trained_model_ckpt/best_model.ckpt")
            x1=inference1.predict(img1,img2)
            inference2 = Inference_prediction2("saved_trained_model_ckpt/contro_model.ckpt")
            x2=inference2.predict(img1,img2)

            # Display the similarity score
            st.markdown("The Disimilarity score between the two images ")
            st.markdown("The Larger the score, Higher the Disimilarity")

            col1, col2= st.columns(2)
            col1.metric(label="Siaseme Network with Binary cross entryopy", value=x1)
            col2.metric(label="Siaseme Network with ContrastiveLoss", value=x2) 

            st.markdown("The Testing Signature is")
            
                        # Displaying text in a green color b
            if (x1+x2)/2 > 50:
                st.markdown(
                '<div style="background-color:red; padding:10px">'
                '<h2 style="color:black;text-align:center;">Forged</h2>'
                '</div>',
                unsafe_allow_html=True)
            else:
                st.markdown(
                '<div style="background-color:lightgreen; padding:10px">'
                '<h2 style="color:black;text-align:center;">Original</h2>'
                '</div>',
                unsafe_allow_html=True)


# Run the Streamlit app
if __name__ == "__main__":
    main()