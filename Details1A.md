# 📄 Document Layout Extraction

## 🎯 Overview

This project implements an intelligent document structure extraction system that automatically analyzes PDF documents to extract hierarchical outlines and structural elements. The system leverages computer vision and machine learning techniques to identify and classify different document components such as titles, headings, abstracts, content sections, and references, then organizes them into a structured JSON format.

🎯 **Primary Goal**: Transform unstructured PDF documents into machine-readable structured data that preserves the document's logical hierarchy and content organization.

## 🔬 Approach

### 🧠 Core Methodology

Our approach combines **object detection**, **optical character recognition (OCR)**, and **clustering algorithms** to create a robust document analysis pipeline. The system processes PDF documents by:

1. 🔄 **Document Conversion**: Converting PDF pages to images for computer vision processing
2. 🎯 **Element Detection**: Using YOLO models to identify and locate document elements
3. 📝 **Text Extraction**: Extracting text content using both PDF text extraction and OCR fallback
4. 📊 **Hierarchical Analysis**: Applying clustering techniques to determine heading levels based on font characteristics
5. 🔧 **Structure Assembly**: Organizing detected elements into a coherent document outline

### 🤖 Dual-Model Architecture

The pipeline employs a **dual-model strategy** to maximize detection accuracy across different document types and scenarios:

#### 🎯 Primary Model (PP-DocLayNet Based)

- **Training Data**: Trained on outputs from the PP-DocLayNet model
- **Purpose**: Optimized for standard academic and professional document layouts
- **Classes**: Specialized in detecting `paragraph_title`, `abstract`, `content`, `table_title`, `reference`, and `doc_title`
- **Advantage**: High accuracy on well-structured documents with consistent formatting

#### 🔄 Backup Model (Handpicked Dataset)

- **Training Data**: Trained on a carefully curated, handpicked dataset
- **Purpose**: Handles edge cases and non-standard document formats
- **Fallback Strategy**: Activates when primary model fails to detect document titles
- **Advantage**: Better generalization to diverse document styles and layouts

This dual-model approach ensures comprehensive coverage across different document scenarios, from standardized academic papers to varied report formats.

## 🚀 Why YOLOv8?

### ⚡ Advantages of YOLOv8 for Document Analysis

1. **Real-time Performance**: YOLOv8's single-pass detection enables fast processing of multi-page documents
2. **High Accuracy**: State-of-the-art object detection capabilities for precise element localization
3. **Versatile Architecture**: Excellent balance between speed and accuracy for document layout analysis
4. **Multi-class Detection**: Efficiently handles multiple document element types in a single inference
5. **Scalability**: Can process documents of varying sizes and complexities
6. **Transfer Learning**: Pre-trained weights allow for effective fine-tuning on document-specific datasets

## 📦 Libraries and Dependencies

### 🔧 Core Libraries

- **ultralytics**: YOLOv8 implementation for object detection
- **PyMuPDF (fitz)**: PDF processing and text extraction
- **OpenCV (cv2)**: Image processing and manipulation
- **EasyOCR**: Optical character recognition fallback
- **NumPy**: Numerical computations and array operations
- **scikit-learn**: K-means clustering for heading level assignment
- **PyTorch**: Deep learning framework backend

## Pipeline Architecture

### Stage 1: Document Preprocessing

- **PDF to Image Conversion**: Each PDF page is converted to a high-resolution image
- **Color Space Normalization**: Handles different color formats (RGB, RGBA, Grayscale)

### Stage 2: Element Detection

- **Primary Detection**: YOLOv8 model identifies document elements with bounding boxes
- **Backup Detection**: Secondary model activates for title detection when primary fails
- **Class Mapping**: Detected elements are mapped to semantic categories

### Stage 3: Text Extraction

- **PDF Text Extraction**: Primary method using PyMuPDF's text extraction
- **OCR Fallback**: EasyOCR processes image regions when PDF text is unavailable
- **Text Cleaning**: Whitespace normalization and text preprocessing

### Stage 4: Hierarchical Analysis

- **Font Size Analysis**: Extracts font characteristics from detected text spans
- **Clustering**: K-means algorithm groups similar text elements by font size
- **Level Assignment**: Clusters are mapped to hierarchical levels (H1, H2, H3)

### Stage 5: Structure Assembly

- **Element Consolidation**: Combines detected elements with hierarchical information
- **Page-wise Organization**: Maintains document page references
- **JSON Export**: Structured output in machine-readable format
-

## 🎯 Use Cases

- **Academic Paper Analysis**: Extract paper structure for literature review tools
- **Report Processing**: Automate corporate report indexing and navigation
- **Content Management**: Generate structured metadata for document repositories
- **Accessibility**: Create navigable outlines for screen readers and assistive technologies

## ✨ Advantages of This Approach

1. **Robustness**: Dual-model strategy handles diverse document types
2. **Accuracy**: Combines multiple text extraction methods for reliability
3. **Scalability**: Batch processing capabilities for large document collections
4. **Flexibility**: Configurable clustering for different document structures
5. **Automation**: Minimal manual intervention required for processing
