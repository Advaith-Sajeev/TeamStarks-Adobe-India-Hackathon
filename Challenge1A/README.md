# 🧩 Challenge 1A – Document Layout Extraction

## 🔧 How to Run

Ensure Docker is installed and working on your system. The solution has been designed to run entirely offline, on a CPU-only environment, and is compliant with the platform constraints specified in the hackathon guidelines.

### 📁 Input/Output Directory Structure

- Place all your input PDF files in the `input/` directory.
- The container will automatically generate the corresponding `.json` files in the `output/` directory.

**Example:**

```
./input/
└── sample.pdf

./output/
└── sample.json # (generated after execution)
```

### 🛠️ Step 1: Build the Docker Image

Run the following command to build the Docker image:

```bash
docker build --platform linux/amd64 -t pdf-processor .
```

### 🚀 Step 2: Execute the Container

Run the following command to start the container and process all PDFs in the input folder:

```bash
 docker run --rm -v ${PWD}/input:/app/input:ro -v ${PWD}/output/repoidentifier/:/app/output --network none pdf-processor
```

### ✅ Expected Behavior

- Automatically process all PDFs from `/input` directory, generating a corresponding `filename.json` in `/output` for each `filename.pdf`
- Output will follow the structure:

```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Heading Text", "page": 1 },
    { "level": "H2", "text": "Subheading Text", "page": 2 },
    ...
  ]
}
```

---
