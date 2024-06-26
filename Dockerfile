# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables to avoid running python in unbuffered mode
ENV PYTHONUNBUFFERED 1

# Create a directory for the app
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies with pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

