# Use a lightweight Python image
FROM python:3.10-slim

# Install system packages required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port (Flask default or your custom port)
EXPOSE 5000

# Start the app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
