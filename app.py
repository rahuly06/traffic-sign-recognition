# app for model
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io
from src.CNN_main import CNNModel  # import your model class

app = FastAPI()

# Load model and weights
model = CNNModel(num_classes=43)  # define your model class
state_dict = torch.load(r"models\traffic_sign_cnn.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Define transforms (same as training)
from torchvision import transforms
transform_pipeline = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.get("/")
def root():
    return {"message": "Welcome!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    class_idx = int(predicted.item())
    return {"class_index": class_idx}
