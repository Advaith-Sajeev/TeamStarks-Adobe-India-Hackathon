# 🧩 Challenge 1B – Multi-Collection PDF Analysis

## Methodology 🚀

This project presents an intelligent, lightweight, CPU-only offline system that converts static PDFs into dynamic, structured, and persona-aware knowledge artifacts. At the heart of this system is a layout-aware Small Language Model (SLM) designed specifically for CPU-only environments.

### Key Features:

- **PDF Extraction**: The system extracts hierarchical elements (titles, headings, etc.) from raw PDFs, which are then used as input for the language model.
- **Semantic Relevance**: The model selects the top 'n' (5) most relevant sections based on their similarity to the query in the embedding space.
- **Summarization**: The selected sections are then summarized by the language model for easy consumption.

### Model Components:

- **Qwen2.5-0.5b (Int8 Quantized)**: Runs on the llama.cpp engine for efficient processing.
- **2x YOLOv8n Models**:

  - YOLO distilled using PP-DocLayout-L.
  - A custom model trained on a hand-picked dataset specialized on outlines.

- **multi-qa-MiniLM-L6-cos-v1**: Utilized for semantic search and ranking of extracted content.
- **K-means Clustering**: Classifies content based on heading levels (H1, H2, H3) for better structure and organization.

The models are optimized to fit within the size constraints defined in Round 1A (200 MB) and Round 1B (1 GB), ensuring efficiency without sacrificing functionality.

## 🔧 How to Run

Ensure Docker is installed and working on your system. The solution has been designed to run entirely offline, on a CPU-only environment, and is compliant with the platform constraints specified in the hackathon guidelines.

## Project Structure

```
Challenge 1B/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                        # South of France guides
│   ├── challenge1b_input.json       # Input configuration
│   └── challenge1b_output.json      # Analysis results
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                        # Acrobat tutorials
│   ├── challenge1b_input.json       # Input configuration
│   └── challenge1b_output.json      # Analysis results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                        # Cooking guides
│   ├── challenge1b_input.json       # Input configuration
│   └── challenge1b_output.json      # Analysis results
└── README.md
```

### 📁 Input/Output Directory Structure

- Place all your input PDF files in the respective `PDFs/` directory inside each collection.
- The system will automatically generate the corresponding `.json` files such as `challenge1b_output.json`, in each collection folder.
- If you just have one set of PDFs put it in a folder Collection 1 folder. Safely ignore the others

### 🛠️ Step 1: Build the Docker Image

Run the following command to build the Docker image:

```bash
docker build -t my-python-app .
```

### 🚀 Step 2: Execute the Container

Run the following command to start the container and process all PDFs in the input folder:

```bash
 docker run -it --rm my-python-app python evaluate.py
```

Here's the corrected and formatted markdown:

### ✅ Expected Behavior

- Automatically process all PDFs from the `PDFs` directory inside each collection, generating a corresponding `filename.json` in the same collection folder for each `filename.pdf`.
- The output will follow the structure shown below:

```
==================== EXECUTION SUMMARY (END‑TO‑END) ====================
Collection         PDFs  Status        Read    Pipeline     Write   End‑to‑End  Extracted  Output/Err
-----------------------------------------------------------------------------------------------------
Collection 1          7  ok           0.00s      20.10s     0.00s       20.10s          5  Sample_PDFs/Collection 1/challenge1b_output.json
Collection 2         15  ok           0.00s      45.05s     0.00s       45.05s          5  Sample_PDFs/Collection 2/challenge1b_output.json
Collection 3          9  ok           0.00s      32.99s     0.00s       32.99s          5  Sample_PDFs/Collection 3/challenge1b_output.json
-----------------------------------------------------------------------
✅ Completed: 3  |  ⚠️ Skipped: 0  |  ❌ Failed: 0  |  PDFs processed: 31
🧮 Grand total wall time (this script): 98.15s
=======================================================================
```

---
