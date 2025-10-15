import argparse
from pathlib import Path

from .utils_io import load_corpus_df, load_queries, load_qrels_robust
from .textproc import tokenize_finance
from .bm25_runner import build_tokens, small_grid, write_trec_run

def build_parser():
    p = argparse.ArgumentParser("BM25 on FiQA (local)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="Evaluate a small (k1,b) grid and write metrics + TREC run")
    g.add_argument("--corpus",  default="data/corpus.jsonl")
    g.add_argument("--queries", default="data/queries.jsonl")
    g.add_argument("--qrels",   default="data/fiqa_qrels_test.tsv")
    g.add_argument("--k", type=int, default=10, help="cutoff for MRR/Recall/nDCG")
    g.add_argument("--grid", default="0.9,0.5;1.2,0.5;1.5,0.5;0.9,0.75;1.2,0.75;1.5,0.75",
                   help="semicolon-separated pairs k1,b (e.g. 0.9,0.5;1.2,0.5)")
    g.add_argument("--out_csv",  default="outputs/metrics/bm25_small_grid.csv")
    g.add_argument("--run_path", default="outputs/runs/run.bm25_sw.trec")

    s = sub.add_parser("search", help="Ad-hoc search for a single query")
    s.add_argument("--corpus", default="data/corpus.jsonl")
    s.add_argument("--k1", type=float, default=0.9)
    s.add_argument("--b",  type=float, default=0.5)
    s.add_argument("--topk", type=int, default=10)
    s.add_argument("--query", required=True, help="your search text")

    return p

def do_grid(args):
    df_corpus = load_corpus_df(Path(args.corpus))
    queries   = load_queries(Path(args.queries))
    gold, _   = load_qrels_robust(Path(args.qrels))

    qids = set(gold.keys())
    queries = [(qid, text) for qid, text in queries if qid in qids]

    print(f"Loaded corpus={len(df_corpus):,}  queries_with_qrels={len(queries):,}")

    doc_ids, tokens = build_tokens(df_corpus, tokenize_finance)

    grid = []
    for pair in args.grid.split(";"):
        k1s, bs = pair.split(",")
        grid.append((float(k1s), float(bs)))

    df = small_grid(tokens, doc_ids, queries, gold, tokenize_finance, grid, K=args.k)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved metrics →", args.out_csv)

    best = df.iloc[0]
    k1, b = float(best.k1), float(best.b)
    Path(args.run_path).parent.mkdir(parents=True, exist_ok=True)
    write_trec_run(tokens, doc_ids, queries, tokenize_finance, k1, b, args.run_path, tag="bm25_sw")
    print("Wrote TREC run →", args.run_path)

def do_search(args):
    from rank_bm25 import BM25Okapi
    import numpy as np
    df_corpus = load_corpus_df(Path(args.corpus))
    doc_ids, tokens = build_tokens(df_corpus, tokenize_finance)

    bm25 = BM25Okapi(tokens, k1=args.k1, b=args.b)
    qtok = tokenize_finance(args.query)
    scores = bm25.get_scores(qtok)

    k = min(args.topk, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    print(f"\nTop-{k} for: {args.query!r}  (k1={args.k1}, b={args.b})\n")
    for rank, i in enumerate(idx, 1):
        doc_id = doc_ids[i]
        score  = float(scores[i])
        text   = df_corpus.iloc[i]["content"]
        snippet = (text[:240] + "…") if len(text) > 240 else text
        print(f"{rank:>2}. Score: {score:.6f}\n    {snippet}\n")

def main():
    args = build_parser().parse_args()
    if args.cmd == "grid":
        do_grid(args)
    elif args.cmd == "search":
        do_search(args)

if __name__ == "__main__":
    main()
