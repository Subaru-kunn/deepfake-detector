from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import os
from tensorflow.keras.models import load_model  # Import the load_model function

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

# Load the model (from .h5 file)
try:
    model = load_model("deepfake_model.h5")  # Load the .h5 model instead of pickle file
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

class ImageRequest(BaseModel):
    image: str

@app.post("/predict/")
async def predict(image_request: ImageRequest):
    try:
        # Decode and resize to 180x180 (match training)
        image_data = base64.b64decode(image_request.image)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize((180, 180))
        image = np.array(image)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Validate shape (1, 180, 180, 3)
        if image.shape != (1, 180, 180, 3):
            raise HTTPException(status_code=400, detail=f"Invalid image shape: {image.shape}")

        # Predict
        prediction = model.predict(image)[0]
        return {"result": "Real" if prediction == 1 else "Fake"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
