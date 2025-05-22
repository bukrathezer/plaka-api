!pip install fastapi uvicorn nest-asyncio pyngrok ultralytics easyocr opencv-python-headless pillow
!pip install python-multipart

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

app = FastAPI()

# Mobil için CORS aç
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli ve OCR'yi yükle
model = YOLO("/content/drive/MyDrive/Arac_Plaka/data/yolov8best.pt")  # kendi modelin
reader = easyocr.Reader(['en'])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    results = model(image_np)

    plates = []
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        if class_name != "plate":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image_np[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result = reader.readtext(gray, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if result:
            plates.append(" ".join(result))

    return {"plates": plates}

!ngrok config add-authtoken 2xSA7zpwvYFAdSsbK0CFNE9LGIw_3DGzzu97c1vTZhcb8xusL

import nest_asyncio
import uvicorn
from pyngrok import ngrok

# Colab için patch
nest_asyncio.apply()

# Tunnel aç (ngrok)
public_url = ngrok.connect(8000)
print("📡 Public URL:", public_url)

# API başlat
uvicorn.run(app, host="0.0.0.0", port=8000)

