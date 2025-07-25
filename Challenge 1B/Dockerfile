# Use an official Python base image
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    cmake \
    ninja-build \
    software-properties-common \
    libomp-dev \
    libatlas-base-dev \
    python3-dev \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
    gcc-11 \
    g++-11 \
    build-essential \
    libstdc++-12-dev

ENV CC=gcc-11
ENV CXX=g++-11

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pillow easyocr

# Install gdown for Google Drive file downloads
RUN pip install --no-cache-dir gdown

RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set NLTK data path and download punkt_tab during build
ENV NLTK_DATA=/usr/local/nltk_data
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt_tab

# --- Test NLTK installation during build ---
RUN python -c "import nltk; nltk.data.path.append('/usr/local/nltk_data'); nltk.data.find('tokenizers/punkt_tab')"

# Create the dummy image using Pillow (100x100 white image)
RUN python -c "from PIL import Image; img = Image.new('RGB', (100, 100), color = (255, 255, 255)); img.save('/app/dummy_image.jpg')"

# Download EasyOCR models during build by using the dummy image
RUN python -c "import easyocr; reader = easyocr.Reader(['en']); reader.readtext('/app/dummy_image.jpg')"

# --- Add Google Drive File Download Step ---
RUN python download_gdrive_folder.py 

CMD ["python", "evaluate.py"]