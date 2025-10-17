import time
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .metrics import mrr_at_k, recall_at_k, ndcg_at_k


def build_tokens(df_corpus, tokenizer):
    """Return parallel lists: doc_ids, tokenized documents.
    Assumes df has columns: _id (str) and content (str).
    """
    df = df_corpus.copy()
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.strip().ne("")]

    doc_ids = df["_id"].astype(str).tolist()
    tokens = [tokenizer(c) for c in df["content"].tolist()]
    return doc_ids, tokens


def evaluate(tokens, doc_ids, queries, gold, tokenizer, Ks=(10,), k1=0.9, b=0.5):
    """Compute MRR/Recall/nDCG at multiple cutoffs.

    Args:
        Ks: iterable of ints (e.g., (1,5,10,100))
    Returns:
        dict with keys like: N, Seconds, MRR@5, Recall@10, nDCG@100, ...
    """
    Ks = tuple(sorted(set(int(k) for k in Ks)))
    bm = BM25Okapi(tokens, k1=k1, b=b)

    sums = {("MRR", K): 0.0 for K in Ks}
    sums.update({("Recall", K): 0.0 for K in Ks})
    sums.update({("nDCG", K): 0.0 for K in Ks})

    t0 = time.time()
    for qid, text in queries:
        qtok = tokenizer(text)
        s = bm.get_scores(qtok)
        kmax = min(max(Ks), len(s))
        if kmax == 0:
            continue
        idx = np.argpartition(s, -kmax)[-kmax:]
        idx = idx[np.argsort(s[idx])[::-1]]
        ranked_ids = [doc_ids[i] for i in idx]
        rel = gold.get(qid, set())
        for K in Ks:
            if K > kmax:
                K_eff = kmax
            else:
                K_eff = K
            pred = ranked_ids[:K_eff]
            sums[("MRR", K)]   += mrr_at_k(pred, rel, K_eff)
            sums[("Recall", K)] += recall_at_k(pred, rel, K_eff)
            sums[("nDCG", K)]  += ndcg_at_k(pred, rel, K_eff)
    dt = time.time() - t0
    n = max(len(queries), 1)

    out = {"N": n, "Seconds": dt}
    for K in Ks:
        out[f"MRR@{K}"]   = sums[("MRR", K)] / n
        out[f"Recall@{K}"] = sums[("Recall", K)] / n
        out[f"nDCG@{K}"]  = sums[("nDCG", K)] / n
    return out


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


def small_grid(tokens, doc_ids, queries, gold, tokenizer, grid, Ks=(10,)):
    """Evaluate multiple (k1,b) pairs; return DataFrame with metrics for all Ks."""
    from rank_bm25 import BM25Okapi

    # Build ONCE (k1,b here are placeholders; overwritten each loop)
    bm = BM25Okapi(tokens, k1=1.0, b=0.5)

    rows = []
    # Which K to display in the console summary
    K_disp = 10 if 10 in set(Ks) else sorted(Ks)[0]

    for k1, b in grid:
        bm.k1, bm.b = float(k1), float(b)

        # Reuse the same index but call evaluate logic with this (k1,b)
        # For efficiency, we inline a minimal copy using bm we just configured:
        sums = {("MRR", K): 0.0 for K in Ks}
        sums.update({("Recall", K): 0.0 for K in Ks})
        sums.update({("nDCG", K): 0.0 for K in Ks})

        t0 = time.time()
        for qid, text in queries:
            qtok = tokenizer(text)
            s = bm.get_scores(qtok)
            kmax = min(max(Ks), len(s))
            if kmax == 0:
                continue
            idx = np.argpartition(s, -kmax)[-kmax:]
            idx = idx[np.argsort(s[idx])[::-1]]
            ranked_ids = [doc_ids[i] for i in idx]
            rel = gold.get(qid, set())
            for K in Ks:
                K_eff = kmax if K > kmax else K
                pred = ranked_ids[:K_eff]
                sums[("MRR", K)]   += mrr_at_k(pred, rel, K_eff)
                sums[("Recall", K)] += recall_at_k(pred, rel, K_eff)
                sums[("nDCG", K)]  += ndcg_at_k(pred, rel, K_eff)
        dt = time.time() - t0
        n = max(len(queries), 1)

        res = {"N": n, "Seconds": dt, "k1": k1, "b": b}
        for K in Ks:
            res[f"MRR@{K}"]   = sums[("MRR", K)] / n
            res[f"Recall@{K}"] = sums[("Recall", K)] / n
            res[f"nDCG@{K}"]  = sums[("nDCG", K)] / n

        rows.append(res)
        print(
            f"k1={k1:.2f}  b={b:.2f} -> "
            f"MRR@{K_disp}={res[f'MRR@{K_disp}']:.4f}  "
            f"R@{K_disp}={res[f'Recall@{K_disp}']:.4f}  "
            f"nDCG@{K_disp}={res[f'nDCG@{K_disp}']:.4f}  "
            f"time={res['Seconds']:.1f}s"
        )

    # Sort by the display K's MRR for convenience
    return pd.DataFrame(rows).sort_values(f"MRR@{K_disp}", ascending=False)
