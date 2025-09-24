# deployment/api/main.py
import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from torchvision import transforms  # Import transforms here
from src.model import TransUNet

# The old import 'from src.data_loader import get_transforms' is no longer needed

app = FastAPI(title="Medical Segmentation API")

# --- Model Loading ---
DEVICE = "cpu"
MODEL_PATH = "transunet_centralized_best.pth"
model = TransUNet(n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Create Inference Transform Pipeline Here ---
# This should match the validation/testing transforms from training,
# but without the random augmentations.
infer_transform = transforms.Compose(
    [
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """Accepts an image, performs segmentation, and returns the mask image."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Use the new inference transform
    input_tensor = infer_transform(image).unsqueeze(0).to(DEVICE)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process mask
    prediction_tensor = output.squeeze().cpu()
    mask_array = (prediction_tensor > 0.5).numpy().astype("uint8") * 255
    mask_image = Image.fromarray(mask_array)

    # Save mask to a byte stream to return it
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
