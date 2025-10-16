import time
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .metrics import mrr_at_k, recall_at_k, ndcg_at_k


def build_tokens(df_corpus, tokenizer):
    """Return parallel lists: doc_ids, tokenized documents.
    Assumes df has columns: _id (str) and content (str).
    """
    # Drop rows with empty content (rare but safer)
    df = df_corpus.copy()
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.strip().ne("")]

    doc_ids = df["_id"].astype(str).tolist()
    tokens = [tokenizer(c) for c in df["content"].tolist()]
    return doc_ids, tokens


def evaluate(tokens, doc_ids, queries, gold, tokenizer, k1=0.9, b=0.5, K=10):
    """Compute MRR@K / Recall@K / nDCG@K over provided queries.
    - queries: list[(qid:str, text:str)] already filtered to have qrels
    - gold: dict[qid] -> set(docid)
    """
    bm = BM25Okapi(tokens, k1=k1, b=b)
    mrr = rec = ndcg = 0.0
    t0 = time.time()
    for qid, text in queries:
        qtok = tokenizer(text)
        s = bm.get_scores(qtok)
        k = min(K, len(s))
        idx = np.argpartition(s, -k)[-k:]
        idx = idx[np.argsort(s[idx])[::-1]]
        pred = [doc_ids[i] for i in idx]
        rel = gold.get(qid, set())
        mrr += mrr_at_k(pred, rel, K)
        rec += recall_at_k(pred, rel, K)
        ndcg += ndcg_at_k(pred, rel, K)
    dt = time.time() - t0
    n = max(len(queries), 1)
    return {"N": n, "MRR@10": mrr / n, "Recall@10": rec / n, "nDCG@10": ndcg / n, "Seconds": dt}


def write_trec_run(tokens, doc_ids, queries, tokenizer, k1, b, out_path, tag="bm25"):
    bm = BM25Okapi(tokens, k1=k1, b=b)
    depth = 1000
    with open(out_path, "w") as out:
        for qid, text in queries:
            qtok = tokenizer(text)
            s = bm.get_scores(qtok)
            k = min(depth, len(s))
            idx = np.argpartition(s, -k)[-k:]
            idx = idx[np.argsort(s[idx])[::-1]]
            for rank, i in enumerate(idx, 1):
                out.write(f"{qid} Q0 {doc_ids[i]} {rank} {s[i]:.6f} {tag}\n")


def small_grid(tokens, doc_ids, queries, gold, tokenizer, grid, K=10):
    from rank_bm25 import BM25Okapi
    # Build ONCE (k1,b placeholders; will be overwritten)
    bm = BM25Okapi(tokens, k1=1.0, b=0.5)

    rows = []
    for k1, b in grid:
        bm.k1, bm.b = float(k1), float(b)  # reuse the same index
        mrr = rec = ndcg = 0.0
        t0 = time.time()
        for qid, text in queries:
            qtok = tokenizer(text)
            s = bm.get_scores(qtok)
            k = min(K, len(s))
            idx = np.argpartition(s, -k)[-k:]
            idx = idx[np.argsort(s[idx])[::-1]]
            pred = [doc_ids[i] for i in idx]
            rel = gold.get(qid, set())
            mrr += mrr_at_k(pred, rel, K)
            rec += recall_at_k(pred, rel, K)
            ndcg += ndcg_at_k(pred, rel, K)
        dt = time.time() - t0
        n = max(len(queries), 1)
        res = {"N": n, "MRR@10": mrr/n, "Recall@10": rec/n, "nDCG@10": ndcg/n,
               "Seconds": dt, "k1": k1, "b": b}
        rows.append(res)
        print(
            f"k1={k1:.2f}  b={b:.2f} -> "
            f"MRR@{K}={res['MRR@10']:.4f}  R@{K}={res['Recall@10']:.4f}  "
            f"nDCG@{K}={res['nDCG@10']:.4f}  time={res['Seconds']:.1f}s"
        )
    return pd.DataFrame(rows).sort_values("MRR@10", ascending=False)
