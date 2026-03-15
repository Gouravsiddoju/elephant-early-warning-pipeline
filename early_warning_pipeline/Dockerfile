# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# (Assuming requirements.txt will be created next)
RUN pip install --no-cache-dir fastapi uvicorn torch joblib numpy pandas pyproj scikit-learn

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run api.py when the container launches
CMD ["python", "api.py"]
