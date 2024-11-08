import streamlit as st
from transformers import pipeline
from PIL import Image

def set_up_page():
    header = st.header("Computer Vision: Wound Image Classifier")
    return header

def set_up_doc_uploader():
    # Use columns to control the width of the file uploader
    col1, col2, col3 = st.columns([2, 1, 1])  # Adjust the relative widths here

    with col1:
        image = st.file_uploader("upload your image and see the type of wound", type=["jpg", "png"])
    return image

def main():

    set_up_page()
    pipe = pipeline("image-classification", model="Heem2/wound-image-classification")

    img = set_up_doc_uploader()

    if img is not None:
        image = Image.open(img)
        result = (pipe(image))
        st.markdown("#### _Wound type:_")
        st.markdown(result[0]['label'])
    
main()