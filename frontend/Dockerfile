# Use Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy frontend files
COPY . .

# Expose Flask on port 5000
EXPOSE 5000

# Run Flask server
CMD ["python", "app.py"]
