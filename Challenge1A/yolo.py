import cv2
import numpy as np
import fitz  # PyMuPDF
import easyocr
from ultralytics import YOLO
import torch
from pathlib import Path
from sklearn.cluster import KMeans
import json
import os

# ---- Default Config ----
DEFAULT_PRIMARY_MODEL_PATH = "./models/pp_yolo/best.pt"
DEFAULT_BACKUP_MODEL_PATH = "./models/new_20k_yolo/best (1).pt"
DEFAULT_NUM_CLUSTERS = 3
DEFAULT_DEVICE = "cpu"

# ---- Class Mapping ----
PRIMARY_CLASSES = {
    0: "paragraph_title",
    4: "abstract",
    5: "content",
    9: "table_title",
    10: "reference",
    11: "doc_title"
}

class PDFProcessor:
    def __init__(self, primary_model_path=None, backup_model_path=None, device="cpu", num_clusters=3):
        """Initialize the PDF processor with model paths and settings"""
        self.primary_model_path = primary_model_path or DEFAULT_PRIMARY_MODEL_PATH
        self.backup_model_path = backup_model_path or DEFAULT_BACKUP_MODEL_PATH
        self.device = device
        self.num_clusters = num_clusters
        torch.backends.cudnn.enabled = False
        
        # Initialize models
        self.primary_model = YOLO(self.primary_model_path).to(self.device)
        self.backup_model = YOLO(self.backup_model_path).to(self.device)
        
        # Initialize EasyOCR reader once
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)

    def extract_text_with_ocr(self, image, bbox):
        """Extract text using OCR from a cropped region"""
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]
        results = self.ocr_reader.readtext(cropped)
        text = " ".join([r[1] for r in results]).strip()
        return text

    def pdf_to_images(self, pdf_path):
        """Convert PDF pages to images"""
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif pix.n == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img
            images.append(img_bgr)
        doc.close()
        return images

    def run_batched_detection(self, images, pdf_path, batch_size=None):
        """Run detection on images in batches for better efficiency"""
        if batch_size is None:
            batch_size = max(1, os.cpu_count() or 8)
        
        doc_items = []
        all_para_titles = []
        title_text = ""
        doc_title_found = False
        pdf_doc = fitz.open(pdf_path)

        # Step 1: Title detection on first 2 pages using PRIMARY model
        title_candidates_by_page = {0: [], 1: []}
        title_pages = min(2, len(images))
        
        if title_pages > 0:
            # Batch process title pages
            title_images = images[:title_pages]
            title_results = self.primary_model(title_images, device=self.device)
            
            for page_num, (img, result) in enumerate(zip(title_images, title_results)):
                boxes = result.boxes
                if boxes is None:
                    continue
                page = pdf_doc.load_page(page_num)
                spans = page.get_text("dict")["blocks"]

                for i in range(len(boxes.cls)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id != 11:  # Only look for doc_title
                        continue
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    bbox = fitz.Rect(*xyxy)
                    matched_texts = []
                    for block in spans:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if fitz.Rect(span["bbox"]).intersects(bbox):
                                    matched_texts.append(span["text"])
                    text = " ".join(matched_texts).strip()
                    if not text:
                        text = self.extract_text_with_ocr(img, xyxy)
                    if text:
                        title_candidates_by_page[page_num].append(text)
                        doc_title_found = True

            # Select title from primary model results
            if doc_title_found:
                if title_candidates_by_page[0]:
                    title_text = " ".join(title_candidates_by_page[0]).strip()
                elif title_candidates_by_page[1]:
                    title_text = " ".join(title_candidates_by_page[1]).strip()

        # Step 2: Fallback to BACKUP model for title if not found
        if not doc_title_found and title_pages > 0:
            backup_title_by_page = {0: [], 1: []}
            title_images = images[:title_pages]
            backup_title_results = self.backup_model(title_images, device=self.device)
            
            for page_num, (img, result) in enumerate(zip(title_images, backup_title_results)):
                boxes = result.boxes
                if boxes is None:
                    continue
                page = pdf_doc.load_page(page_num)
                spans = page.get_text("dict")["blocks"]

                for i in range(len(boxes.cls)):
                    cls_id = int(boxes.cls[i].item())
                    label = self.backup_model.names.get(cls_id, "").lower()
                    if "title" not in label:
                        continue
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    bbox = fitz.Rect(*xyxy)
                    matched_texts = []
                    for block in spans:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if fitz.Rect(span["bbox"]).intersects(bbox):
                                    matched_texts.append(span["text"])
                    text = " ".join(matched_texts).strip()
                    if not text:
                        text = self.extract_text_with_ocr(img, xyxy)
                    if text:
                        backup_title_by_page[page_num].append(text)

            if backup_title_by_page[0]:
                title_text = " ".join(backup_title_by_page[0]).strip()
            elif backup_title_by_page[1]:
                title_text = " ".join(backup_title_by_page[1]).strip()

        # Step 3: Process ALL pages in batches for content detection
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            batch_page_nums = list(range(batch_start, batch_end))
            
            try:
                # Run batch inference
                batch_results = self.primary_model(batch_images, device=self.device)
            except Exception as e:
                if logs:
                    print(f"❌ Batch inference failed for pages {batch_start}-{batch_end-1}: {e}")
                continue
            
            # Process each result in the batch
            for page_offset, (img, result) in enumerate(zip(batch_images, batch_results)):
                page_num = batch_page_nums[page_offset]
                boxes = result.boxes
                if boxes is None:
                    continue
                
                page = pdf_doc.load_page(page_num)
                spans = page.get_text("dict")["blocks"]

                for i in range(len(boxes.cls)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id == 11:  # Skip doc_title as we already processed it
                        continue
                    label = PRIMARY_CLASSES.get(cls_id)
                    if label is None:
                        continue
                    
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    bbox = fitz.Rect(*xyxy)
                    matched_texts = []
                    
                    for block in spans:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if fitz.Rect(span["bbox"]).intersects(bbox):
                                    matched_texts.append((span["text"], span["size"]))
                    
                    text = " ".join([t for t, _ in matched_texts]).strip()
                    if not text:
                        continue
                    
                    avg_font_size = np.mean([s for _, s in matched_texts]) if matched_texts else 0
                    top_y = xyxy[1]  # y1 coordinate for vertical sorting

                    if cls_id == 0:  # paragraph_title
                        doc_items.append({
                            "type": "paragraph_title",
                            "text": text,
                            "page": page_num,
                            "font_size": avg_font_size,
                            "y": top_y
                        })
                        all_para_titles.append((text, avg_font_size, page_num, top_y))
                    elif cls_id in [4, 5, 10]:  # abstract, content, reference
                        doc_items.append({
                            "type": label,
                            "text": label.capitalize(),
                            "page": page_num
                        })

        pdf_doc.close()
        return doc_items, all_para_titles, title_text

    def assign_heading_levels(self, paragraph_items):
        """Assign heading levels using clustering with Y-coordinate sorting"""
        if not paragraph_items:
            return []
        # Sort by page then y (top to bottom)
        paragraph_items = sorted(paragraph_items, key=lambda x: (x[2], x[3]))  # (page, y)
        font_sizes = np.array([[fs] for _, fs, _, _ in paragraph_items])
        
        # Handle case where we have fewer items than clusters
        n_clusters = min(self.num_clusters, len(paragraph_items))
        if n_clusters == 0:
            return []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(font_sizes)
        labels = kmeans.labels_
        center_map = dict(zip(range(n_clusters), kmeans.cluster_centers_.flatten()))
        sorted_clusters = sorted(center_map, key=lambda k: -center_map[k])
        level_map = {sorted_clusters[i]: f"H{i+1}" for i in range(n_clusters)}
        
        outline = []
        for (text, _, page, _), label in zip(paragraph_items, labels):
            outline.append({
                "level": level_map[label],
                "text": text,
                "page": page
            })
        return outline

    def postprocess_items(self, doc_items, outline):
        """Final outline postprocessing"""
        final_outline = outline.copy()
        for item in doc_items:
            if item["type"] in ["abstract", "content", "reference"]:
                final_outline.append({
                    "level": "H2",
                    "text": item["text"],
                    "page": item["page"]
                })
        return sorted(final_outline, key=lambda x: x["page"])

    def save_to_json(self, title, outline, path, logs=True):
        """Save results to JSON file"""
        output = {
            "title": title,
            "outline": outline
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        if logs:
            print(f"✅ JSON saved to {path}")

    def process_single_pdf(self, pdf_path, output_path, batch_size=None, logs=True):
        """Process a single PDF file"""
        try:
            if logs:
                print(f"📄 Processing: {Path(pdf_path).name}")
            
            images = self.pdf_to_images(str(pdf_path))
            
            # Use batched detection
            doc_items, paragraph_titles, title_text = self.run_batched_detection(
                images, str(pdf_path), batch_size
            )
            
            outline = self.assign_heading_levels(paragraph_titles)
            final_outline = self.postprocess_items(doc_items, outline)

            self.save_to_json(title_text, final_outline, str(output_path), logs)
            return True
            
        except Exception as e:
            if logs:
                print(f"❌ Failed to process {Path(pdf_path).name}: {e}")
            return False


# ---- Public API Functions ----
def process_all_pdfs(pdf_folder, output_folder, primary_model_path=None, backup_model_path=None,
                    device="cpu", num_clusters=3, batch_size=None, logs=True, show_plots=False):
    """
    Process all PDFs in a folder
    
    Args:
        pdf_folder (str/Path): Folder containing PDF files
        output_folder (str/Path): Folder to save JSON outputs
        primary_model_path (str): Path to primary YOLO model
        backup_model_path (str): Path to backup YOLO model
        device (str): Device to run inference on ('cpu' or 'cuda')
        num_clusters (int): Number of clusters for heading level assignment
        batch_size (int): Batch size for inference (None for auto)
        logs (bool): Whether to print logs
        show_plots (bool): Whether to show plots (not implemented in original)
    """
    # Initialize processor
    processor = PDFProcessor(
        primary_model_path=primary_model_path,
        backup_model_path=backup_model_path,
        device=device,
        num_clusters=num_clusters
    )
    
    # Create output directory
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Calculate optimal batch size if not provided
    if batch_size is None:
        batch_size = max(1, os.cpu_count() or 8)
    
    if logs:
        print(f"🚀 Using batch size: {batch_size}")
    
    # Process all PDFs
    pdf_folder = Path(pdf_folder)
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        if logs:
            print(f"⚠️ No PDF files found in {pdf_folder}")
        return
    
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        output_path = output_folder / (pdf_file.stem + ".json")
        success = processor.process_single_pdf(
            pdf_file, output_path, batch_size, logs
        )
        if success:
            successful += 1
        else:
            failed += 1
    
    if logs:
        print(f"🎉 Processing complete! ✅ {successful} successful, ❌ {failed} failed")


def process_single_pdf_file(pdf_path, output_path, primary_model_path=None, backup_model_path=None,
                           device="cpu", num_clusters=3, batch_size=None, logs=True):
    """
    Process a single PDF file
    
    Args:
        pdf_path (str/Path): Path to PDF file
        output_path (str/Path): Path to save JSON output
        primary_model_path (str): Path to primary YOLO model
        backup_model_path (str): Path to backup YOLO model
        device (str): Device to run inference on ('cpu' or 'cuda')
        num_clusters (int): Number of clusters for heading level assignment
        batch_size (int): Batch size for inference (None for auto)
        logs (bool): Whether to print logs
    
    Returns:
        bool: True if successful, False otherwise
    """
    processor = PDFProcessor(
        primary_model_path=primary_model_path,
        backup_model_path=backup_model_path,
        device=device,
        num_clusters=num_clusters
    )
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    return processor.process_single_pdf(pdf_path, output_path, batch_size, logs)


# ---- Entry Point (for backward compatibility) ----
def main():
    """Original main function for backward compatibility"""
    PDF_DIR = "./Sample_PDFs/Collection 2/PDFs"
    OUTPUT_DIR = "./output_jsons"
    
    process_all_pdfs(
        pdf_folder=PDF_DIR,
        output_folder=OUTPUT_DIR,
        logs=True
    )


if __name__ == "__main__":
    main()