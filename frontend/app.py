from flask import Flask, render_template, request
import requests
import base64
import os
import time

app = Flask(__name__)

# FastAPI backend URL
BACKEND_URL = "https://deepfake-detector-backend-nfkc.onrender.com/predict/"


def post_with_retry(url, payload, retries=3, delay=1):
    """
    Sends a POST request with retry logic for handling 429 errors.
    Retries with exponential backoff (1s → 2s → 4s).
    """
    for i in range(retries):
        response = requests.post(url, json=payload)
        if response.status_code == 429:
            print(f"Rate limit hit, retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        else:
            return response
    return response  # return last attempt


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            # Read image and encode to Base64
            image_bytes = file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Send image as JSON payload
            payload = {"image": image_base64}

            try:
                response = post_with_retry(BACKEND_URL, payload)

                if response.status_code == 200:
                    result = response.json().get("result", "Unknown")
                    # Display uploaded image
                    image_data = f"data:image/jpeg;base64,{image_base64}"
                elif response.status_code == 429:
                    result = "Too many requests, please try again later."
                else:
                    result = f"Backend error: {response.status_code}"

            except Exception as e:
                result = f"Error: {str(e)}"

    return render_template("index.html", result=result, image_data=image_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
