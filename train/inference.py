# save as encode_bundle_bge.py
# pip install -U sentence-transformers torch
import json, numpy as np, torch
from sentence_transformers import SentenceTransformer

BUNDLE = "bundle.json"
MODEL  = 'ENTER MODEL PATH'
OUT_NPY = "embeddings.npy"       # matrix: [1 + #docs, dim]
OUT_TXT = "embedding_order.txt"  # lines describing each row

# --- load bundle
b = json.load(open(BUNDLE, encoding="utf-8"))
query = b["query"]
docs  = b["documents"]           # [{"doc_id":..., "text":..., "label":...}, ...]

# --- stable order: query, then green, then yellow, then red (preserve JSON order within each label)
prio = {"green": 0, "yellow": 1, "red": 2}
ordered = sorted(list(enumerate(docs)), key=lambda x: (prio.get(x[1]["label"], 2), x[0]))
ordered_docs = [d for _, d in ordered]

# --- texts to encode (apply BGE query instruction to the query)
def qfmt(x): return x
texts = [qfmt(query)] + [d["text"] for d in ordered_docs]

# --- load model & encode
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)
model.normalize_embeddings = True  # BGE works best with normalized embeddings

with torch.inference_mode():
    embs = model.encode(texts, batch_size=64, convert_to_tensor=True)  # shape: [N, dim]

# --- save outputs
np.save(OUT_NPY, embs.cpu().numpy())
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("0\tQUERY\n")
    for i, d in enumerate(ordered_docs, start=1):
        f.write(f"{i}\t{d['doc_id']}\t{d.get('label','')}\n")

print(f"Saved matrix to {OUT_NPY} with shape {tuple(embs.shape)}")
print(f"Row order written to {OUT_TXT}")