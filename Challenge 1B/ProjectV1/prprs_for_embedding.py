from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
from typing import Mapping, Union, Tuple, Dict, Any
import argparse
import json

__all__ = [
    "merge_folder_to_single_json",
    "write_merged_json",
    "merge_processed_docs"
]

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def merge_folder_to_single_json(input_dir: Union[str, Path]) -> "OrderedDict[str, Dict[str, str]]":
    """
    Scan `input_dir` for *_outline.json and *_sections.json files and merge them
    into a single OrderedDict of sections, indexed as docXX_secYY.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    outline_files = sorted(input_dir.glob("*_outline.json"))

    master: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    doc_counter = 0

    for outline_path in outline_files:
        base = outline_path.stem.replace("_outline", "")
        sections_path = input_dir / f"{base}_sections.json"
        if not sections_path.exists():
            continue

        doc_counter += 1
        merged, _ = _process_document(doc_counter, outline_path, sections_path)
        master.update(merged)

    return master


def write_merged_json(
    data: Mapping[str, Dict[str, Any]],
    output_file: Union[str, Path],
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """
    Write the merged data to a JSON file.
    """
    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def merge_processed_docs(processed_jsons: list[dict]) -> "OrderedDict[str, Dict[str, str]]":
    """
    Merge a list of in-memory processed document dicts into one big OrderedDict of sections.

    Each dict must have:
    {
        "doc_title": str,
        "outline": List[dict],
        "sections": List[dict] | Dict[str, str]
    }

    Returns:
        OrderedDict[str, Dict[str, str]]
    """
    master: "OrderedDict[str, Dict[str, str]]" = OrderedDict()

    for doc_idx, doc_data in enumerate(processed_jsons, start=1):
        outline_json = {
            "title": doc_data.get("doc_title", ""),
            "outline": doc_data.get("outline", [])
        }
        sections_json = doc_data.get("sections", {})

        sections_dict = _normalize_sections(sections_json)
        headers = _extract_outline_headers_in_order(outline_json)

        doc_prefix = f"doc{doc_idx:02d}"
        for sec_idx, header in enumerate(headers, 1):
            key = f"{doc_prefix}_sec{sec_idx:02d}"
            master[key] = {
                "doc_title": outline_json["title"],
                "section_header": header,
                "section_body": (sections_dict.get(header) or "").strip()
            }

    return master

# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_sections(sections_json) -> "OrderedDict[str, str]":
    """
    Normalize sections to: {section_header: section_body}
    """
    if isinstance(sections_json, dict):
        out = OrderedDict()
        for k, v in sections_json.items():
            out[str(k)] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return out

    if isinstance(sections_json, list):
        out = OrderedDict()
        for i, item in enumerate(sections_json, 1):
            if not isinstance(item, dict):
                out[f"Section {i}"] = str(item)
                continue
            header = (
                item.get("section_header")
                or item.get("header")
                or item.get("title")
                or item.get("name")
                or f"Section {i}"
            )
            body = (
                item.get("section_body")
                or item.get("body")
                or item.get("text")
                or ""
            )
            out[str(header)] = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        return out

    return OrderedDict([("_whole_file", json.dumps(sections_json, ensure_ascii=False))])

def _extract_outline_headers_in_order(outline_json: dict) -> list[str]:
    """
    Extract outline headers in order.
    """
    outline_entries = outline_json.get("outline") or []
    headers = []
    for e in outline_entries:
        txt = e.get("text") or e.get("title") or e.get("header")
        if isinstance(txt, str) and txt.strip():
            headers.append(txt.strip())
    return headers

def _process_document(
    doc_idx: int,
    outline_path: Path,
    sections_path: Path
) -> Tuple["OrderedDict[str, Dict[str, str]]", int]:
    outline_json  = _load_json(outline_path)
    sections_json = _load_json(sections_path)

    doc_title = ""
    if isinstance(outline_json.get("title"), str):
        doc_title = outline_json.get("title", "").strip()

    sections_dict = _normalize_sections(sections_json)
    headers = _extract_outline_headers_in_order(outline_json)

    doc_prefix = f"doc{doc_idx:02d}"
    output: "OrderedDict[str, Dict[str, str]]" = OrderedDict()

    for sec_idx, header in enumerate(headers, 1):
        key = f"{doc_prefix}_sec{sec_idx:02d}"
        output[key] = {
            "doc_title": doc_title,
            "section_header": header,
            "section_body": (sections_dict.get(header) or "").strip()
        }

    return output, len(output)

# ---------------------------------------------------------------------
# CLI (optional)
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge *_outline.json + *_sections.json into a single JSON.")
    p.add_argument("-i", "--input_dir", required=True, help="Folder that holds *_outline.json and *_sections.json")
    p.add_argument("-o", "--output_file", required=True, help="Path to combined output JSON")
    return p.parse_args()

def _main():
    args = _parse_args()
    data = merge_folder_to_single_json(args.input_dir)
    write_merged_json(data, args.output_file)
    print(f"Done. Documents merged: {len({k.split('_sec')[0] for k in data.keys()})}, total sections: {len(data)}")
    print(f"Wrote: {args.output_file}")

if __name__ == "__main__":
    _main()
