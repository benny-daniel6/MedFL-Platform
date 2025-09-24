# deployment/app/app.py
import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

st.set_page_config(layout="wide")
st.title("🩺 MedFL: AI-Powered Medical Image Segmentation")

st.sidebar.header("About")
st.sidebar.info(
    "This dashboard demonstrates a medical image segmentation model "
    "trained using Federated Learning. Upload a brain MRI scan to "
    "see the AI identify and segment the tumor region."
)

API_URL = "http://127.0.0.1:8000/segment"

uploaded_file = st.file_uploader(
    "Choose a brain MRI image...", type=["jpg", "png", "tif"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    original_image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.header("Original Image")
        # Changed use_column_width to use_container_width
        st.image(original_image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        try:
            response = requests.post(API_URL, files=files)
            response.raise_for_status()

            mask_image = Image.open(io.BytesIO(response.content))

            # Resize the mask to the same size as the original image
            mask_image = mask_image.resize(original_image.size)

            # Overlay mask on original image
            original_rgba = original_image.convert("RGBA")

            # Create a red overlay with transparency
            red_overlay = Image.new("RGBA", original_rgba.size, (255, 0, 0, 0))
            red_overlay.paste((255, 0, 0, 128), mask=mask_image.convert("L"))

            # Alpha composite the images
            overlay_image = Image.alpha_composite(original_rgba, red_overlay)

            with col2:
                st.header("AI Segmentation")
                # Changed use_column_width to use_container_width
                st.image(
                    overlay_image,
                    use_container_width=True,
                    caption="Red overlay indicates the predicted tumor region.",
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API. Ensure it is running. Error: {e}")
