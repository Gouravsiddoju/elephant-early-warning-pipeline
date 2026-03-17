# Use official Python runtime as a parent image
FROM python:3.14-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn torch==2.10.0 joblib==1.5.3 numpy==2.4.2 pandas==2.3.3 scikit-learn==1.8.0 pyproj

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run api.py when the container launches
CMD ["python", "api.py"]
