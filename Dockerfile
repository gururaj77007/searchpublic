# Use an official Python runtime as a parent image
FROM --platform=linux/amd64  python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for pandas, MongoDB, and TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libhdf5-serial-dev \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    python3-dev \
    curl \
    && apt-get clean

# Install pip dependencies
RUN pip install --upgrade pip
RUN pip install flask pandas tensorflow tqdm pinecone-client symspellpy sentence-transformers scikit-learn numpy pymongo tf-keras

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for Flask API Key
ENV API_KEY="ff7d854b-3c80-4f7c-84fd-0781022e293a"
ENV PINECONE_CLOUD="aws"
ENV PINECONE_REGION="us-east-1"

# Run the Flask app
CMD ["python", "app.py"]
