# Create the base image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on 
EXPOSE 5000

# Command to run the application
CMD ["python3", "app.py"]