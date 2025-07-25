# Set the base image to Python 3.10
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install system dependencies for OpenCV and clean up to reduce image size
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean

# Copy requirements.txt first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout=600 --retries=5

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pillow easyocr

# Copy the application code and models
COPY process_pdfs.py yolo.py ./
COPY models/ ./models/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Create the dummy image using Pillow (100x100 white image)
RUN python -c "from PIL import Image; img = Image.new('RGB', (100, 100), color = (255, 255, 255)); img.save('/app/dummy_image.jpg')"

# Download EasyOCR models during build by using the dummy image
RUN python -c "import easyocr; reader = easyocr.Reader(['en']); reader.readtext('/app/dummy_image.jpg')"


# Set up the default command to run the PDF processing script
CMD ["python", "process_pdfs.py"]
