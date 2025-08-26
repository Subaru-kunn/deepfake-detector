from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import os
from tensorflow.keras.models import load_model  

# -------------------
# Rate Limiter Imports
# -------------------
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

# --------------------
# Setup Rate Limiter
# --------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too Many Requests"})


# --------------------
# Load Model
# --------------------
try:
    model = load_model("deepfake_model.h5")  # Loading model
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


# --------------------
# Request Model
# --------------------
class ImageRequest(BaseModel):
    image: str


# --------------------
# Predict Endpoint
# --------------------
@app.post("/predict/")
@limiter.limit("5/minute")  # Allow max 5 requests per minute per IP
async def predict(request: Request, image_request: ImageRequest):
    try:
        # Decode and resize to 180x180 (match training)
        image_data = base64.b64decode(image_request.image)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize((180, 180))
        image = np.array(image)

        # Adding batch dimension
        image = np.expand_dims(image, axis=0)

        # Validating shape
        if image.shape != (1, 180, 180, 3):
            raise HTTPException(status_code=400, detail=f"Invalid image shape: {image.shape}")

        # Predict
        prediction = model.predict(image)[0]
        predicted_class = "Fake" if prediction >= 0.5 else "Real"
        return {"result": predicted_class}

    except Exception as e:
        return {"error": str(e)}


# --------------------
# Health Check Endpoint
# --------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}
