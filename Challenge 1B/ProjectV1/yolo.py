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
from collections import defaultdict

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

# Surrogate range for text cleaning (from original code)
SURROGATE_RANGE = dict.fromkeys(range(0xD800, 0xE000))

def remove_surrogates(s: str) -> str:
    """Remove surrogate characters from text (from original code)"""
    return s.translate(SURROGATE_RANGE)

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
        """Convert PDF pages to images with page numbers (compatible with original)"""
        doc = fitz.open(pdf_path)
        images = []
        for i, page in enumerate(doc):
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
            # Include page number and file path like original
            images.append((img_bgr, i, pdf_path))
        doc.close()
        return images

    def classify_font_size(self, size, sorted_sizes):
        """Classify font size into H1, H2, H3 levels (from original code)"""
        if len(sorted_sizes) == 0:
            return "H3"
        h1 = sorted_sizes[0]
        h2 = sorted_sizes[1] if len(sorted_sizes) > 1 else h1 - 1
        h3 = sorted_sizes[2] if len(sorted_sizes) > 2 else h2 - 1
        if abs(size - h1) < 0.5:
            return "H1"
        elif abs(size - h2) < 0.5:
            return "H2"
        else:
            return "H3"

    def extract_section_text(self, pdf_doc, page_num, current_bbox, next_bbox, current_text):
        """Extract text content for a section (similar to original logic)"""
        page = pdf_doc.load_page(page_num)
        cur_bbox = fitz.Rect(*current_bbox)
        next_bbox_rect = fitz.Rect(*next_bbox) if next_bbox else None
        
        blocks = page.get_text("dict")["blocks"]
        collected_text = []
        
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_rect = fitz.Rect(span["bbox"])
                    # Text should be below current header
                    if span_rect.y0 > cur_bbox.y1:
                        # But above next header if it exists
                        if next_bbox_rect and span_rect.y0 >= next_bbox_rect.y0:
                            continue
                        span_text = span["text"].strip()
                        if span_text and span_text != current_text:
                            collected_text.append(remove_surrogates(span_text))
        
        return " ".join(collected_text).strip()

    def run_batched_detection(self, all_images, batch_size=None, logs=False):
        """Run detection on images in batches, returning data compatible with original format"""
        if batch_size is None:
            batch_size = max(1, os.cpu_count() or 8)
        
        # Group images by PDF
        pdf_groups = defaultdict(list)
        for img_data in all_images:
            img, page_num, pdf_path = img_data
            pdf_groups[pdf_path].append((img, page_num))
        
        all_results = []
        
        for pdf_path, images_data in pdf_groups.items():
            if logs:
                print(f"📄 Processing: {Path(pdf_path).name}")
            
            # Extract images and page numbers
            images = [img for img, _ in images_data]
            page_nums = [page_num for _, page_num in images_data]
            
            doc_items = []
            all_para_titles = []
            title_text = ""
            doc_title_found = False
            pdf_doc = fitz.open(pdf_path)
            all_font_sizes = []
            section_items = []
            section_bboxes = defaultdict(list)
            title_texts = []

            # Step 1: Title detection on first 2 pages using PRIMARY model
            title_pages = min(2, len(images))
            
            if title_pages > 0:
                title_images = images[:title_pages]
                title_results = self.primary_model(title_images, device=self.device)
                
                for page_idx, (img, result) in enumerate(zip(title_images, title_results)):
                    page_num = page_nums[page_idx]
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
                        font_sizes = []
                        
                        for block in spans:
                            if "lines" not in block:
                                continue
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    if fitz.Rect(span["bbox"]).intersects(bbox):
                                        matched_texts.append(span["text"])
                                        font_sizes.append(span["size"])
                        
                        text = " ".join(matched_texts).strip()
                        if not text:
                            text = self.extract_text_with_ocr(img, xyxy)
                        
                        if text:
                            title_texts.append(text)
                            doc_title_found = True
                            all_font_sizes.extend(font_sizes)

            # Step 2: Fallback to BACKUP model for title if not found
            if not doc_title_found and title_pages > 0:
                title_images = images[:title_pages]
                backup_title_results = self.backup_model(title_images, device=self.device)
                
                for page_idx, (img, result) in enumerate(zip(title_images, backup_title_results)):
                    page_num = page_nums[page_idx]
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
                        font_sizes = []
                        
                        for block in spans:
                            if "lines" not in block:
                                continue
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    if fitz.Rect(span["bbox"]).intersects(bbox):
                                        matched_texts.append(span["text"])
                                        font_sizes.append(span["size"])
                        
                        text = " ".join(matched_texts).strip()
                        if not text:
                            text = self.extract_text_with_ocr(img, xyxy)
                        
                        if text:
                            title_texts.append(text)
                            all_font_sizes.extend(font_sizes)

            # Step 3: Process ALL pages in batches for content detection
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch_images = images[batch_start:batch_end]
                batch_page_nums = [page_nums[i] for i in range(batch_start, batch_end)]
                
                try:
                    batch_results = self.primary_model(batch_images, device=self.device)
                except Exception as e:
                    if logs:
                        print(f"❌ Batch inference failed for pages {batch_start}-{batch_end-1}: {e}")
                    continue
                
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
                        font_sizes = []
                        
                        for block in spans:
                            if "lines" not in block:
                                continue
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    if fitz.Rect(span["bbox"]).intersects(bbox):
                                        matched_texts.append(span["text"])
                                        font_sizes.append(span["size"])
                        
                        text = " ".join(matched_texts).strip()
                        if not text:
                            continue
                        
                        text = remove_surrogates(text)
                        avg_font_size = np.mean(font_sizes) if font_sizes else None
                        all_font_sizes.extend(font_sizes)

                        if cls_id == 0:  # paragraph_title (section header)
                            section_items.append({
                                "text": text,
                                "page": page_num,
                                "size": avg_font_size,
                                "bbox": tuple(xyxy)
                            })
                            section_bboxes[page_num].append({
                                "text": text,
                                "bbox": tuple(xyxy)
                            })
                            
                            if logs:
                                print(f"[{os.path.basename(pdf_path)} - Page {page_num}] Section-header: {text}")

            # Build final result in original format
            outline = []
            seen = set()
            sorted_sizes = sorted(set(all_font_sizes), reverse=True)

            for item in section_items:
                key = (item["text"].lower(), item["page"])
                if key in seen:
                    continue
                seen.add(key)
                level = self.classify_font_size(item["size"], sorted_sizes) if item["size"] else "H3"
                outline.append({
                    "level": level,
                    "text": item["text"],
                    "page": item["page"]
                })

            # Extract section text content
            section_texts = {}
            for page_num, headers in section_bboxes.items():
                headers = sorted(headers, key=lambda h: h['bbox'][1])  # Sort by y coordinate
                for i, current in enumerate(headers):
                    cur_text = current['text']
                    cur_bbox = current['bbox']
                    next_bbox = headers[i + 1]['bbox'] if i + 1 < len(headers) else None
                    
                    section_content = self.extract_section_text(pdf_doc, page_num, cur_bbox, next_bbox, cur_text)
                    section_texts[cur_text] = section_content

            # Get final title
            title = "  ".join(dict.fromkeys(title_texts)).strip()

            result = {
                "doc_title": title,
                "outline": outline,
                "sections": section_texts
            }

            pdf_doc.close()
            all_results.append(result)
            
            if logs:
                print(f"✅ Processed {Path(pdf_path).name}: {len(outline)} sections found")

        return all_results

    def save_to_json(self, title, outline, sections, base_path, logs=True):
        """Save results to JSON files (original format)"""
        # Save outline
        outline_path = f"{base_path}_outline.json"
        with open(outline_path, "w", encoding="utf-8") as f:
            json.dump({"title": title, "outline": outline}, f, indent=2, ensure_ascii=False)
        
        # Save sections
        sections_path = f"{base_path}_sections.json"
        with open(sections_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)
        
        if logs:
            print(f"✅ Saved outline: {outline_path}")
            print(f"📝 Saved section body text: {sections_path}")


# ---- Main Function Compatible with Original ----
def process_all_pdfs(pdf_folder, output_folder=None, logs=False, show_plots=False, return_data=False,
                    primary_model_path=None, backup_model_path=None, device="cpu", num_clusters=3, 
                    batch_size=None):
    """
    Process all PDFs in a folder - Compatible with original function signature
    
    This function maintains the exact same interface as the original code but uses
    the improved detection models internally.
    """
    # Create output folder if needed
    if output_folder and not return_data:
        os.makedirs(output_folder, exist_ok=True)

    # Initialize processor
    processor = PDFProcessor(
        primary_model_path=primary_model_path,
        backup_model_path=backup_model_path,
        device=device,
        num_clusters=num_clusters
    )

    # Collect all images from all PDFs (original format)
    all_images = []
    pdf_doc_map = {}

    for fname in os.listdir(pdf_folder):
        if not fname.lower().endswith(".pdf"):
            continue
        full_path = os.path.join(pdf_folder, fname)
        images = processor.pdf_to_images(full_path)  # Returns (img, page_num, pdf_path) tuples
        all_images.extend(images)
        pdf_doc_map[full_path] = fitz.open(full_path)

    if batch_size is None:
        batch_size = os.cpu_count() or 8

    if logs:
        print(f"\n📚 Loaded {len(pdf_doc_map)} PDFs with {len(all_images)} pages. Using batch size: {batch_size}")

    # Process all images
    results = processor.run_batched_detection(all_images, batch_size, logs)

    # Close all PDF documents
    for doc in pdf_doc_map.values():
        doc.close()

    # Handle output
    if return_data:
        return results
    else:
        # Save to files (original behavior)
        for i, result in enumerate(results):
            # Get the PDF filename for this result
            pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
            if i < len(pdf_files):
                base_name = os.path.splitext(pdf_files[i])[0]
                base_path = os.path.join(output_folder, base_name)
                processor.save_to_json(
                    result["doc_title"], 
                    result["outline"], 
                    result["sections"], 
                    base_path, 
                    logs
                )

    return None


# ---- Additional API Functions ----
def process_single_pdf_file(pdf_path, output_path, primary_model_path=None, backup_model_path=None,
                           device="cpu", num_clusters=3, batch_size=None, logs=True):
    """Process a single PDF file"""
    processor = PDFProcessor(
        primary_model_path=primary_model_path,
        backup_model_path=backup_model_path,
        device=device,
        num_clusters=num_clusters
    )
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the PDF
    images = processor.pdf_to_images(str(pdf_path))
    results = processor.run_batched_detection([images], batch_size, logs)
    
    if results:
        result = results[0]
        base_path = str(Path(output_path).with_suffix(''))
        processor.save_to_json(
            result["doc_title"], 
            result["outline"], 
            result["sections"], 
            base_path, 
            logs
        )
        return True
    return False


# ---- Entry Point ----
def main():
    """Main function for standalone usage"""
    PDF_DIR = "./Sample_PDFs/Collection 2/PDFs"
    OUTPUT_DIR = "./output_jsons"
    
    process_all_pdfs(
        pdf_folder=PDF_DIR,
        output_folder=OUTPUT_DIR,
        logs=True
    )


if __name__ == "__main__":
    main()