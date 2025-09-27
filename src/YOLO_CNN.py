# %%
import os
os.chdir(r"C:\Users\rahul\OneDrive\Desktop\Rahul\Study\Projects\traffic-sign-recognition")

# %% [markdown]
# YOLO import

# %%
from ultralytics import YOLO
model_path = "models/runs/detect/train17/weights/best.pt"
yolo_model = YOLO(model_path)

# %% [markdown]
# CNN model import

# %%
# import CNN model
from CNN_main import CNNModel
import torch

model_CNN = CNNModel(num_classes=43)
model_CNN.load_state_dict(torch.load('models/traffic_sign_cnn.pth', map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_CNN.to(device).eval()

# Preprocess the image for CNN
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),    
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define human-readable labels
traffic_sign_names_dict = {
    0: "speed limit 20", 1: "speed limit 30", 2: "speed limit 50", 3: "speed limit 60",
    4: "speed limit 70", 5: "speed limit 80", 6: "end of speed limit 80", 7: "speed limit 100",
    8: "speed limit 120", 9: "no overtaking", 10: "no overtaking for trucks", 11: "priority at next intersection",
    12: "priority road", 13: "give way", 14: "stop", 15: "no traffic both ways", 16: "no trucks",
    17: "no entry", 18: "danger", 19: "bend left", 20: "bend right", 21: "bend", 22: "uneven road",
    23: "slippery road", 24: "road narrows", 25: "construction", 26: "traffic signal",
    27: "pedestrian crossing", 28: "school crossing", 29: "cycles crossing", 30: "snow", 31: "animals",
    32: "restriction ends 50", 33: "restriction ends 80", 34: "end of no overtaking",
    35: "end of no overtaking for trucks", 36: "turn right ahead", 37: "turn left ahead",
    38: "ahead only", 39: "go right", 40: "go left", 41: "keep right", 42: "keep left"
}

# %% [markdown]
# Inference loop

# %%
import cv2
import numpy as np

def process_frame(frame):
    results = yolo_model(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor_img = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_CNN(tensor_img)
                _, predicted = torch.max(outputs, 1)
                class_id = predicted.item()

            # Map class_id to human-readable label
            class_label = traffic_sign_names_dict.get(class_id, "Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{class_label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame

# %% [markdown]
# Trial run

# %%
cap = cv2.VideoCapture("video/screengrab_fromYoutube_night.mp4")

if not cap.isOpened():
    print("❌ Could not open video file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)
    cv2.imshow("YOLO + CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()