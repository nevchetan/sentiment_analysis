# Use a Python image that is stable and slim
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
# This is done first to leverage Docker's build caching
COPY requirements.txt .
# --no-cache-dir reduces the final image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (code, model, tokenizer)
COPY . /app

# Hugging Face Spaces mandates using port 7860 for web services
EXPOSE 7860

# Command to run your Flask app using Gunicorn
# --bind 0.0.0.0:7860: Sets the host and port
# --workers 1: Keeps memory use low for the free tier
# --timeout 120: IMPORTANT! Gives Keras/TensorFlow time to load the model (2 minutes)
# app:app: Specifies that the Flask app object named 'app' is in the 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]