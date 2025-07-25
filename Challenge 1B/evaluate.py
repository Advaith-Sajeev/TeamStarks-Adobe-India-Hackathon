# runner_end_to_end.py
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from v1 import run_pipeline_parallel  # your pipeline function

# ================= Configuration =================
BASE_DIR = Path(".//Sample_PDFs")                 # parent folder containing "Collection 1", "Collection 2", ...
COLLECTIONS_GLOB = "Collection *"                # pattern to find collection folders
PREFERRED_INPUT_JSON = "challenge1b_input.json"  # default input file name per collection
OUTPUT_JSON_NAME = "challenge1b_output.json"     # output file name written inside each collection

# Models
EMBEDDING_MODEL_PATH = "./models/multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL_PATH = "./models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q8_0.gguf"

# Pipeline knobs
TRIM_TOP_N = 5
TOP_K = 5

# If True, fsync after writing JSON so timing includes disk flush
FSYNC_OUTPUT = True


# ================= Helpers =================
def find_collections(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob(COLLECTIONS_GLOB) if p.is_dir()])


def find_input_json(collection_dir: Path) -> Optional[Path]:
    # Prefer the canonical file name; otherwise pick any *input*.json
    candidate = collection_dir / PREFERRED_INPUT_JSON
    if candidate.exists():
        return candidate
    matches = sorted(collection_dir.glob("*input*.json"))
    return matches[0] if matches else None


def list_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted([p for p in pdf_dir.glob("*.pdf")] + [p for p in pdf_dir.glob("*.PDF")])


def ensure_documents_in_input(input_data: Dict[str, Any], pdfs: List[Path]) -> Dict[str, Any]:
    """
    If input JSON lacks a `documents` list, populate it from the folder so metadata is accurate.
    """
    if not input_data.get("documents"):
        input_data = dict(input_data)  # shallow copy
        input_data["documents"] = [{"filename": p.name} for p in pdfs]
    return input_data


def pretty_seconds(s: float) -> str:
    return f"{s:.2f}s"


def write_json_end_to_end(path: Path, payload: Dict[str, Any], fsync: bool = True) -> None:
    """
    Write JSON and optionally fsync so timing includes kernel flush.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


# ================= Runner =================
def main():
    collections = find_collections(BASE_DIR)
    if not collections:
        print(f"❌ No collections found under: {BASE_DIR.resolve()}")
        return

    grand_start = time.perf_counter()
    summary_rows = []

    print(f"🔎 Found {len(collections)} collection(s) under {BASE_DIR.resolve()}")

    for coll in collections:
        pdf_dir = coll / "PDFs"
        if not pdf_dir.is_dir():
            print(f"\n➡️  {coll.name}: skipped (no 'PDFs' folder)")
            summary_rows.append({
                "collection": coll.name,
                "num_pdfs": 0,
                "status": "skipped",
                "read_s": 0.0,
                "pipeline_s": 0.0,
                "write_s": 0.0,
                "end_to_end_s": 0.0,
                "output": None,
                "error": "No PDFs folder",
                "extracted_sections": "-",
            })
            continue

        input_json_path = find_input_json(coll)
        if not input_json_path:
            print(f"\n➡️  {coll.name}: skipped (no input JSON found)")
            summary_rows.append({
                "collection": coll.name,
                "num_pdfs": len(list_pdfs(pdf_dir)),
                "status": "skipped",
                "read_s": 0.0,
                "pipeline_s": 0.0,
                "write_s": 0.0,
                "end_to_end_s": 0.0,
                "output": None,
                "error": "No input JSON",
                "extracted_sections": "-",
            })
            continue

        pdfs = list_pdfs(pdf_dir)
        print(f"\n📂 Processing {coll.name} — PDFs: {len(pdfs)}  |  Input: {input_json_path.name}")

        coll_start = time.perf_counter()
        # ---- READ PHASE ----
        t0 = time.perf_counter()
        with input_json_path.open("r", encoding="utf-8") as f:
            input_data = json.load(f)
        input_data = ensure_documents_in_input(input_data, pdfs)
        t1 = time.perf_counter()

        # ---- PIPELINE PHASE ----
        try:
            result = run_pipeline_parallel(
                input_data=input_data,
                pdf_folder=str(pdf_dir),
                llm_model_path=LLM_MODEL_PATH,
                st_model_path=EMBEDDING_MODEL_PATH,
                trim_top_n=TRIM_TOP_N,
                k=TOP_K,
                llm_kwargs={
                # init
                "n_ctx": 4096,
                "n_threads": None,
                "chat_format": "chatml",
                "verbose": False,
                "n_batch": 1024,
                # inference - improved settings
                "max_tokens": 512,
                # "temperature": 0.1,  # Lower for more consistent JSON
                # "top_p": 0.9,
                # "top_k": 40,
                # "seed": 42,
                },
                
            )
            t2 = time.perf_counter()
        except Exception as e:
            t2 = time.perf_counter()
            coll_end = time.perf_counter()
            read_s = t1 - t0
            pipeline_s = t2 - t1
            end_to_end_s = coll_end - coll_start
            print(f"❌ {coll.name} failed after {pretty_seconds(end_to_end_s)}: {e}")
            summary_rows.append({
                "collection": coll.name,
                "num_pdfs": len(pdfs),
                "status": "error",
                "read_s": read_s,
                "pipeline_s": pipeline_s,
                "write_s": 0.0,
                "end_to_end_s": end_to_end_s,
                "output": None,
                "error": str(e),
                "extracted_sections": "-",
            })
            continue

        # ---- WRITE PHASE ----
        try:
            output_path = coll / OUTPUT_JSON_NAME
            write_start = time.perf_counter()
            write_json_end_to_end(output_path, result, fsync=FSYNC_OUTPUT)
            write_end = time.perf_counter()
            print(f"📁 Output saved to {output_path}")
        except Exception as e:
            write_end = time.perf_counter()
            print(f"⚠️  {coll.name} output write error: {e}")

        coll_end = time.perf_counter()

        read_s = t1 - t0
        pipeline_s = t2 - t1
        write_s = write_end - write_start
        end_to_end_s = coll_end - coll_start

        print(f"⏱️  {coll.name} end-to-end in {pretty_seconds(end_to_end_s)} "
              f"(read {pretty_seconds(read_s)} | pipeline {pretty_seconds(pipeline_s)} | write {pretty_seconds(write_s)})")

        summary_rows.append({
            "collection": coll.name,
            "num_pdfs": len(pdfs),
            "status": "ok",
            "read_s": read_s,
            "pipeline_s": pipeline_s,
            "write_s": write_s,
            "end_to_end_s": end_to_end_s,
            "output": str(output_path),
            "error": "",
            "extracted_sections": len(result.get("extracted_sections", [])),
        })

    total_time = time.perf_counter() - grand_start

    # ================= Summary =================
    print("\n==================== EXECUTION SUMMARY (END‑TO‑END) ====================")
    header = (f"{'Collection':<18} {'PDFs':>4}  {'Status':<8}  "
              f"{'Read':>8}  {'Pipeline':>10}  {'Write':>8}  {'End‑to‑End':>11}  {'Extracted':>9}  {'Output/Err'}")
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        output = row.get("output") or row.get("error", "")
        print(f"{row['collection']:<18} {row['num_pdfs']:>4}  {row['status']:<8}  "
              f"{pretty_seconds(row['read_s']):>8}  {pretty_seconds(row['pipeline_s']):>10}  "
              f"{pretty_seconds(row['write_s']):>8}  {pretty_seconds(row['end_to_end_s']):>11}  "
              f"{str(row.get('extracted_sections','-')):>9}  {output}")
    print("-----------------------------------------------------------------------")
    ok = sum(1 for r in summary_rows if r["status"] == "ok")
    skipped = sum(1 for r in summary_rows if r["status"] == "skipped")
    failed = sum(1 for r in summary_rows if r["status"] == "error")
    total_pdfs = sum(r["num_pdfs"] for r in summary_rows if r["status"] != "skipped")
    print(f"✅ Completed: {ok}  |  ⚠️ Skipped: {skipped}  |  ❌ Failed: {failed}  |  PDFs processed: {total_pdfs}")
    print(f"🧮 Grand total wall time (this script): {pretty_seconds(total_time)}")
    print("=======================================================================\n")


if __name__ == "__main__":
    main()
