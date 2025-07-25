# Adobe India Hackathon 2025 – Connecting the Dots 🚀

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

## Challenge Solutions

### [Challenge 1a: PDF Processing Solution](./Challenge1A/README.md)

Basic PDF processing with Docker containerization and structured data extraction.

### [Challenge 1b: Multi-Collection PDF Analysis](./Challenge%201B/approach_explanation.md)

Advanced persona-based content analysis across multiple document collections.

---

**Note**: Each challenge directory contains detailed documentation and implementation details. Please refer to the individual README files for comprehensive information about each solution.
