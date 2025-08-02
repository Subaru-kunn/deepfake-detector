from flask import Flask, render_template, request
import requests
import base64

app = Flask(__name__)

# FastAPI backend URL
BACKEND_URL = "http://backend:8000/predict/"  # Docker service name

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            # Read the image and encode it in Base64
            image_bytes = file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Send image as JSON payload
            payload = {"image": image_base64}

            try:
                response = requests.post(BACKEND_URL, json=payload)
                if response.status_code == 200:
                    result = response.json().get("result", "Unknown")
                    # Display the uploaded image
                    image_data = f"data:image/jpeg;base64,{image_base64}"
                else:
                    result = f"Backend error: {response.status_code}"
            except Exception as e:
                result = f"Error: {str(e)}"

    return render_template("index.html", result=result, image_data=image_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
