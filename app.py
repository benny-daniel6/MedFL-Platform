# app.py (for Hugging Face Spaces)
import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from torchvision import transforms
from src.model import TransUNet

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("🩺 MedFL: AI-Powered Medical Image Segmentation")


# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the model and cache it for the app's lifecycle."""
    device = "cpu"
    model_path = "transunet_centralized_best.pth"
    model = TransUNet(n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


model = load_model()

# --- Preprocessing ---
infer_transform = transforms.Compose(
    [
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# --- UI Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This dashboard demonstrates a medical image segmentation model "
    "trained using Federated Learning. The model (a TransUNet) analyzes "
    "brain MRI scans to identify and segment tumor regions."
)

# --- Main UI ---
uploaded_file = st.file_uploader(
    "Choose a brain MRI image...", type=["jpg", "png", "tif"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.header("Original Image")
        st.image(original_image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Preprocess and predict
        input_tensor = infer_transform(original_image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)

        # Post-process mask
        prediction_tensor = output.squeeze()
        mask_array = (prediction_tensor > 0.5).numpy().astype("uint8") * 255
        mask_image = Image.fromarray(mask_array).resize(original_image.size)

        # Create overlay
        original_rgba = original_image.convert("RGBA")
        red_overlay = Image.new("RGBA", original_rgba.size, (255, 0, 0, 0))
        red_overlay.paste((255, 0, 0, 128), mask=mask_image.convert("L"))
        overlay_image = Image.alpha_composite(original_rgba, red_overlay)

        with col2:
            st.header("AI Segmentation")
            st.image(
                overlay_image,
                use_container_width=True,
                caption="Red overlay indicates the predicted tumor region.",
            )
