# ===== ProjectV1/getTop5.py  (REPLACE ENTIRE FILE) ===========================
import json, nltk, torch, numpy as np
from pathlib import Path
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
nltk.download("punkt", quiet=True)

# --------------------------------------------------------------------------- #
# 1.  TRIM SECTIONS TO TOP-N CENTRAL SENTENCES                                #
# --------------------------------------------------------------------------- #
def trim_sections_to_central_sentences(
    all_sections: dict,
    model: SentenceTransformer,
    top_n: int = 5,
    batch_size: int = 64,
) -> dict:
    """
    Returns {section_id: {..., 'section_body': trimmed_text}}
    """
    trimmed = {}
    batched_sent_lists, meta = [], []   # meta[i] == (section_id, original_record, sentences)

    for sid, rec in all_sections.items():
        body = rec.get("section_body", "").strip()
        if not body:          # skip empty
            continue
        sents = sent_tokenize(body)
        if not sents:
            continue
        batched_sent_lists.append(sents)
        meta.append((sid, rec, sents))

    # ---- bulk-embed every sentence ----------------------------------------
    flat_sentences = [s for group in batched_sent_lists for s in group]
    sent_embs       = model.encode(flat_sentences, batch_size=16, convert_to_numpy=True)

    # ---- walk back through the meta to build trimmed bodies ----------------
    cursor = 0
    for sid, rec, sents in meta:
        n = len(sents)
        cur_embs = sent_embs[cursor: cursor + n]
        cursor  += n

        if n <= top_n:
            trimmed_body = " ".join(sents)
        else:
            centroid   = np.mean(cur_embs, axis=0)
            sims       = cosine_similarity([centroid], cur_embs)[0]
            keep_idxs  = sims.argsort()[-top_n:][::-1]          # top-N by similarity
            top_sents  = [sents[i] for i in sorted(keep_idxs)]  # keep original order
            trimmed_body = " ".join(top_sents)

        trimmed[sid] = {
            **rec,                       # doc_title, header, etc. (if you stored them)
            "section_body": trimmed_body # overwrite body with trimmed version
        }
    return trimmed

# --------------------------------------------------------------------------- #
# 2.  EMBEDDING + kNN RETRIEVAL                                               #
# --------------------------------------------------------------------------- #
def embed_sections(sections: dict, model: SentenceTransformer):
    keys, texts = zip(*[(k, v["section_body"]) for k, v in sections.items() if v["section_body"]])
    embs = model.encode(list(texts), convert_to_tensor=True)
    return list(keys), embs

def retrieve_top_k_sections(
    query: str,
    trimmed_sections: dict,
    model: SentenceTransformer,
    k: int = 5,
):
    query_emb           = model.encode(query, convert_to_tensor=True)
    section_ids, embs   = embed_sections(trimmed_sections, model)
    sims                = util.cos_sim(query_emb, embs)[0]
    topk                = torch.topk(sims, k=min(k, len(sims)))

    results = []
    for rank, idx in enumerate(topk.indices.tolist(), 1):
        sid   = section_ids[idx]
        rec   = trimmed_sections[sid]
        score = float(topk.values[rank-1])
        results.append({
            "rank"      : rank,
            "section_id": sid,
            "doc_id"    : sid.split("_")[0],
            "header"    : rec.get("section_header", ""),
            "body"      : rec["section_body"],
            "score"     : score
        })
    return results
# =========================================================================== #
