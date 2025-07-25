# 🧠 PDF Summarization Pipeline – Methodology Overview

This project implements an **end-to-end, in-memory PDF processing and summarization pipeline** powered by modern AI models. The core goal is to extract meaningful content from PDFs, identify the most relevant sections for a specified task, and generate short, structured summaries using a locally hosted LLM. Below is a detailed description of the methodology used:

## 🔬 Methodology

### 1. 🤖 **LLM Initialization (Asynchronous)**

The pipeline begins by **loading a local Large Language Model (LLM)** in a background thread using the `llama_cpp` library. This model is loaded in-memory only, without writing any files. Parameters like context length, threading, batching, and inference settings are customizable via `llm_kwargs`. This asynchronous loading ensures that other stages can start concurrently, reducing overall latency.

### 2. 📄 **PDF Content Extraction**

Using a custom `process_all_pdfs()` method (integrated with YOLO-based document layout parsing), the system processes each PDF to extract structured content such as:

- Document title
- Section headers and bodies
- Page numbers

This is done **entirely in-memory**, avoiding file I/O.

### 3. 🔗 **Section Merging**

The extracted documents are merged into a single dictionary structure using `merge_processed_docs()`. Each section retains its metadata (document ID, page number, header), forming a clean, indexed representation of all textual content.

### 4. ✂️ **Text Trimming via SBERT**

A Sentence-BERT model (MiniLM) is loaded locally to compute sentence embeddings. Each section is **trimmed to its top-N most central sentences**, preserving only the most representative parts of the text. This trimming is guided by semantic centrality within the section.

### 5. 🔍 **Top-K Section Retrieval**

Given a user-defined `task` (e.g., "summarize safety issues"), the same SBERT model computes similarity between the task query and each trimmed section. The **top-K most relevant sections** are selected for summarization.

### 6. 📝 **Prompt Construction for LLM**

The selected sections are formatted into a well-structured prompt, including:

- Section rank
- Document name
- Page number
- Header and trimmed content

The prompt also includes explicit instructions to the LLM: produce concise, task-relevant summaries (1–2 sentences each, under 75 words), formatted as a valid JSON list.

### 7. 🎯 **LLM Summarization**

Once the LLM is fully loaded, the prompt is sent for inference. The response is parsed as JSON. If parsing fails, the raw section body is used as a fallback.

### 8. ✅ **Final Output Construction**

The final result includes:

- Metadata (input file names, persona, task, timestamp)
- Extracted section details (document name, title, rank, page number)
- LLM-generated summaries

The system ensures summaries end at the last period and are concise, clean, and structured.

## ⭐ Key Features

- **💾 In-Memory Processing**: No temporary files created during execution
- **⚡ Parallel Processing**: Background LLM loading while document processing occurs
- **🔧 Flexible Configuration**: Customizable model parameters and processing settings
- **🛡️ Robust Error Handling**: Graceful fallbacks for JSON parsing and content processing
- **📊 Page Number Normalization**: Handles both zero-based and one-based page indexing systems

## 🏗️ Technical Architecture

The pipeline combines multiple AI technologies: computer vision for document structure analysis, transformer models for semantic understanding, and large language models for intelligent summarization. This multi-modal approach ensures comprehensive document understanding and high-quality output generation suitable for various analytical tasks.
