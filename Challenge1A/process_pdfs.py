import os
import json
from pathlib import Path
from yolo import process_all_pdfs

def process_pdfs():
    """Process PDFs from /app/input and save to /app/output (Docker-compatible)"""
    # Use Docker-friendly paths that match volume mounts
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify directories exist and are accessible
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if not output_dir.exists():
        print(f"Error: Could not create output directory: {output_dir}")
        return
    
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    
    # Check if input directory has any PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        print(f"Please place your PDF files in the {input_dir.resolve()} folder")
        
        # List what files are actually there for debugging
        all_files = list(input_dir.iterdir())
        if all_files:
            print(f"Files found in directory: {[f.name for f in all_files if f.is_file()]}")
        else:
            print("Directory is empty")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) in {input_dir}")
    
    # Process all PDFs
    process_all_pdfs(
        pdf_folder=input_dir,       # Folder with PDFs (app/inputs)
        output_folder=output_dir,   # Where to save output JSONs (app/outputs)
        logs=True,                  # Enable logging
        show_plots=False            # Disable plots
    )
    

def process_single_pdf(pdf_filename):
    """Process a single PDF by filename from /app/input (Docker-compatible)"""
    # Use Docker-friendly paths that match volume mounts
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = input_dir / pdf_filename
    if not pdf_path.exists():
        print(f"File not found: {pdf_path.resolve()}")
        return False
    
    output_path = output_dir / (pdf_path.stem + ".json")
    
    # Import the single file processor
    from yolo import process_single_pdf_file
    
    print(f"Processing: {pdf_path.resolve()}")
    print(f"Output to: {output_path.resolve()}")
    
    return process_single_pdf_file(
        pdf_path=pdf_path,
        output_path=output_path,
        logs=True
    )


if __name__ == "__main__":
    print("Starting PDF processing...")
    process_pdfs() 
    print("Completed PDF processing!")