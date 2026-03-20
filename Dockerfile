FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install git and any other necessary system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Install the local package
RUN pip install -e .