import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import atexit

from ProjectV1.yolo import process_all_pdfs
from ProjectV1.prprs_for_embedding import merge_processed_docs
from ProjectV1.getTop5 import (
    trim_sections_to_central_sentences,
    retrieve_top_k_sections,
)
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# --------------------------- Globals ---------------------------------
llm_model_loaded = threading.Event()
llm_model: Optional[Llama] = None


def cleanup_llm():
    """Properly cleanup the LLM model"""
    global llm_model
    if llm_model is not None:
        try:
            llm_model.close()
        except Exception as e:
            print(f"Warning: Error during LLM cleanup: {e}")
        finally:
            llm_model = None


# Register cleanup function
atexit.register(cleanup_llm)


def _truncate_to_last_period(text: str) -> str:
    """
    Return text truncated at the last '.' (inclusive). If no period exists,
    return the stripped text unchanged.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    idx = text.rfind(".")
    return text[: idx + 1].strip() if idx != -1 else text.strip()


def load_llm_model(
    llm_path: str,
    n_threads: Optional[int] = None,
    n_ctx: int = 4096,
    chat_format: str = "chatml",
    verbose: bool = False,
    n_batch: int = 512,
):
    """
    Load the local GGUF LLM (e.g., Qwen) on a background thread.
    This function does not create or write any files.
    """
    global llm_model

    if n_threads is None:
        n_threads = os.cpu_count() or 8

    try:
        llm_model_local = Llama(
            model_path=llm_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            chat_format=chat_format,
            verbose=verbose,
            n_batch=n_batch,
        )

        # publish after fully constructed to avoid races
        globals()["llm_model"] = llm_model_local
        llm_model_loaded.set()
        print("✅ LLM model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading LLM model: {e}")
        llm_model_loaded.set()  # Set event even on failure to avoid hanging


def _normalize_page_number(page_no: Optional[int], assume_zero_based: bool) -> int:
    """
    Return a 1-based page number.
    - If page_no is None, default to 1.
    - If upstream numbers are 0-based and assume_zero_based=True, add 1.
    """
    if page_no is None:
        return 1
    return page_no + 1 if assume_zero_based else page_no


def _clean_json_response(text: str) -> str:
    """Clean and extract JSON from LLM response"""
    text = text.strip()
    
    # Find JSON array start and end
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    
    # If no array found, try to find object
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    
    return text


def run_pipeline_parallel(
    input_data: Dict[str, Any],
    pdf_folder: str,
    llm_model_path: str,
    st_model_path: str = "models/multi-qa-MiniLM-L6-cos-v1",
    trim_top_n: int = 5,
    k: int = 5,
    # Set this True if your upstream tools (YOLO/merge/trim/retrieve) produce 0-based page indices.
    assume_zero_based_pages: bool = False,
    # Single, combined kwargs dict (init + inference). Only specific keys are used.
    llm_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline (fully in-memory):
      1) Start loading LLM in background (no files written).
      2) Process PDFs -> outlines/sections in memory.
      3) Merge into one section-indexed dict in memory.
      4) Trim each section to its most central sentences (in memory).
      5) SBERT retrieval for top-K sections (in memory).
      6) Build prompt and run LLM summarization (in memory).
      7) Post-process each summary by truncating at the last period.
      8) Return structured JSON output (not written to disk).

    The `llm_kwargs` dict combines init and inference options, limited to:
      Init keys:  n_ctx, n_threads, chat_format, verbose, n_batch
      Infer keys: max_tokens, temperature, top_p, top_k, seed
    Unknown keys are ignored.
    """
    documents = input_data["documents"]
    persona = input_data["persona"]["role"]
    task = input_data["job_to_be_done"]["task"]
    query = task

    # Allowed keys in this version
    INIT_KEYS = {"n_ctx", "n_threads", "chat_format", "verbose", "n_batch"}
    INFER_KEYS = {"max_tokens", "temperature", "top_p", "top_k", "seed"}

    # Defaults if not provided
    if llm_kwargs is None:
        llm_kwargs = {
            # init defaults
            "n_ctx": 4096,
            "n_threads": None,      # will be resolved to os.cpu_count()
            "chat_format": "chatml",
            "verbose": False,
            "n_batch": 1024,
            # inference defaults
            "max_tokens": 512,  # Increased for better JSON generation
            "temperature": 0.1,   # Lower temperature for more consistent JSON
            "top_p": 0.9,
            "top_k": 40,
            "seed": 42,
        }

    # Split into init vs inference configs (ignore unknown keys)
    init_conf = {k: v for k, v in llm_kwargs.items() if k in INIT_KEYS}
    infer_conf = {k: v for k, v in llm_kwargs.items() if k in INFER_KEYS}

    # 1) Load the LLM in the background
    print("⚙️ Loading LLM model in a background thread (in-memory only)...")
    threading.Thread(
        target=load_llm_model,
        args=(llm_model_path,),
        kwargs=init_conf,
        daemon=True,
    ).start()

    # 2) Process PDFs (YOLO inference) - IN MEMORY
    print("🚀 Starting PDF processing (no file outputs)...")
    processed_docs = process_all_pdfs(
        pdf_folder=pdf_folder,
        output_folder=None,   # ensure nothing is written
        logs=False,
        show_plots=False,
        return_data=True      # return structures instead of saving
    )

    # 3) Merge processed docs - IN MEMORY
    print("🔗 Merging processed documents (in memory)...")
    merged_data = merge_processed_docs(processed_docs)

    # 4) Load Sentence-Transformers model and trim sections - IN MEMORY
    print(f"⚙️ Loading SBERT model from: {st_model_path} (no caching writes)")
    st_model = SentenceTransformer(st_model_path)

    print(f"✂️ Trimming sections to top-{trim_top_n} central sentences (in memory)...")
    trimmed_sections = trim_sections_to_central_sentences(
        merged_data, st_model, top_n=trim_top_n, batch_size=64
    )

    # 5) Wait for LLM to finish loading
    print("⏳ Waiting for the LLM to finish loading...")
    llm_model_loaded.wait()
    
    if llm_model is None:
        raise RuntimeError("Failed to load LLM model")
    
    print("✅ LLM loaded. Proceeding with retrieval and summarization...")

    # 6) Retrieve top-k relevant sections using SBERT - IN MEMORY
    print("🔎 Retrieving top-k relevant sections (in memory)...")
    top_sections = retrieve_top_k_sections(query, trimmed_sections, st_model, k=k)

    # 7) Format retrieved sections for prompt + bookkeeping - IN MEMORY
    extracted_sections = []
    prompt_sections = []

    def _doc_id(sec: Dict[str, Any]) -> str:
        # best-effort doc identifier
        return sec.get("doc_id") or sec.get("document") or sec.get("doc_title") or "UNKNOWN_DOCUMENT"

    for idx, sec in enumerate(top_sections, start=1):
        raw_page_no = sec.get("page_number")
        page_no = _normalize_page_number(raw_page_no, assume_zero_based_pages)

        extracted_sections.append({
            "document": _doc_id(sec),
            "section_title": sec.get("header", "") or sec.get("section_header", ""),
            "importance_rank": idx,
            "page_number": page_no,
        })

        # Clean content for better processing
        content = sec.get('body', '') or sec.get('section_body', '')
        content = content.replace('\n', ' ').replace('\r', ' ').strip()
        
        prompt_sections.append(
            f"[Section {idx}] Document: {_doc_id(sec)}\n"
            f"Title: {sec.get('header','') or sec.get('section_header','')}\n"
            f"Page: {page_no}\n"
            f"Content: {content}"
        )

    print("📄 Extracted sections for LLM prompt:")
    print(prompt_sections)
    print()

    # Improved prompt for better JSON generation
    prompt = f"""You are an {persona}. Your task is to **{task}** using the content provided below.

Below are the most relevant sections extracted from different documents:

{chr(10).join(prompt_sections)}

Please analyze each section and create concise summaries. Follow these requirements exactly:

1. Write a maximum 2-sentence summary for each section (under 75 words)
2. End each summary with a period
3. Focus on relevance to the task: "{task}"
4. Return ONLY a valid JSON array with this exact structure:

[
  {{
    "document": "document_name",
    "refined_text": "Your concise summary here.",
    "page_number": page_number
  }}
]

Do not include any other text, explanations, or formatting. Only return the JSON array."""

    # 8) Run LLM summarization - IN MEMORY
    print("🤖 Running LLM inference...")
    try:
        response = llm_model(prompt, **infer_conf)
        raw_text = response["choices"][0]["text"].strip()
        
        print(f"🤖 LLM Raw Response: {raw_text}...")  # Debug output
        
        # Clean and extract JSON
        clean_json = _clean_json_response(raw_text)
        
    except Exception as e:
        print(f"❌ Error during LLM inference: {e}")
        raw_text = ""
        clean_json = ""

    # 9) Parse JSON or fallback - IN MEMORY, with post-processing at last period
    try:
        if clean_json:
            subsection_analysis = json.loads(clean_json)
            
            # Validate and fix the structure
            if isinstance(subsection_analysis, list):
                for item in subsection_analysis:
                    if isinstance(item, dict):
                        # Ensure required keys exist
                        if "refined_text" not in item:
                            item["refined_text"] = "Summary not available."
                        if "document" not in item:
                            item["document"] = "UNKNOWN"
                        if "page_number" not in item:
                            item["page_number"] = 1
                        
                        # Truncate at last period
                        item["refined_text"] = _truncate_to_last_period(item.get("refined_text", ""))
            else:
                raise ValueError("Response is not a list")
                
        else:
            raise ValueError("No clean JSON found")
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ JSON parsing failed ({e}), using fallback...")
        # Fallback: create summaries from the original sections
        subsection_analysis = []
        for idx, sec in enumerate(top_sections):
            content = sec.get("body", "") or sec.get("section_body", "")
            # Truncate content to reasonable length for fallback
            if len(content) > 150:
                content = content
            
            subsection_analysis.append({
                "document": _doc_id(sec),
                "refined_text": _truncate_to_last_period(content),
                "page_number": _normalize_page_number(sec.get("page_number"), assume_zero_based_pages),
            })

    final_output = {
        "metadata": {
            "input_documents": [doc.get("filename") or doc.get("path") or str(doc) for doc in documents],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.utcnow().isoformat(),
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis,
    }

    return final_output


def main():
    """Main function with proper error handling"""
    try:
        # Only reads the input JSON; does NOT write any files.
        input_json_path = "./Sample_PDFs/Collection 1/challenge1b_input.json"
        with open(input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        result = run_pipeline_parallel(
            input_data=input_data,
            pdf_folder="./Sample_PDFs/Collection 2/PDFs",
            llm_model_path="./models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q8_0.gguf",
            st_model_path="models/multi-qa-MiniLM-L6-cos-v1",
            trim_top_n=5,
            k=5,
            assume_zero_based_pages=True,
            llm_kwargs={
                # init
                "n_ctx": 4096,
                "n_threads": None,
                "chat_format": "chatml",
                "verbose": False,
                "n_batch": 512,
                # inference - improved settings
                "max_tokens": 512,
                # "temperature": 0.1,  # Lower for more consistent JSON
                # "top_p": 0.9,
                # "top_k": 40,
                # "seed": 42,
            },
        )

        # Print only; no file is saved.
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        cleanup_llm()


if __name__ == "__main__":
    main()