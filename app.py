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

traffic_sign_names = [
    "Speed limit 20 km/h",           # 0
    "Speed limit 30 km/h",           # 1
    "Speed limit 50 km/h",           # 2
    "Speed limit 60 km/h",           # 3
    "Speed limit 70 km/h",           # 4
    "Speed limit 80 km/h",           # 5
    "End of speed limit 80 km/h",    # 6
    "Speed limit 100 km/h",          # 7
    "Speed limit 120 km/h",          # 8
    "No passing",                    # 9
    "No passing for vehicles over 3.5 tons", # 10
    "Right-of-way at intersection",  # 11
    "Priority road",                 # 12
    "Yield",                         # 13
    "Stop",                          # 14
    "No vehicles",                   # 15
    "Vehicles over 3.5 tons prohibited", # 16
    "No entry",                       # 17
    "General caution",                # 18
    "Dangerous curve left",           # 19
    "Dangerous curve right",          # 20
    "Double curve",                   # 21
    "Bumpy road",                     # 22
    "Slippery road",                  # 23
    "Road narrows on the right",      # 24
    "Road work",                      # 25
    "Traffic signals",                # 26
    "Pedestrians",                    # 27
    "Children crossing",              # 28
    "Bicycles crossing",              # 29
    "Turn right ahead",               # 30
    "Turn left ahead",                # 31
    "Ahead only",                     # 32
    "Go straight or right",           # 33
    "Go straight or left",            # 34
    "Keep right",                     # 35
    "Keep left",                      # 36
    "Roundabout mandatory",           # 37
    "End of no passing",              # 38
    "End of no passing by vehicles over 3.5 tons", # 39
    "Class 40 placeholder",           # 40 → replace with actual name
    "Class 41 placeholder",           # 41 → replace with actual name
    "Class 42 placeholder"            # 42 → replace with actual name
]


class_to_idx = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7,
                '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14,
                '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21,
                '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28,
                '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35,
                '41': 36, '42': 37, '5': 38, '6': 39, '7': 40, '8': 41, '9': 42}

# Flip mapping: idx -> folder name
idx_to_folder = {v: k for k, v in class_to_idx.items()}

# Final map: model index -> traffic sign name
class_map = {idx: traffic_sign_names[int(folder_name)] for idx, folder_name in idx_to_folder.items()}

# to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Api endpoints
@app.get("/")
def root():
    return {"message": "Welcome to Traffic Sign Recognition API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform_pipeline(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = int(predicted.item())
        class_name = class_map[class_idx]  # human-readable label

    return {"class_index": class_idx, "class_name": class_name}