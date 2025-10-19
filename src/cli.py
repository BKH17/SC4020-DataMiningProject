import argparse
from pathlib import Path

from .utils_io import load_corpus_df, load_queries, load_qrels_robust
from .textproc import tokenize_finance
from .bm25_runner import build_tokens, small_grid, write_trec_run


def data_paths(root: Path, dataset: str, split: str):
    base = root / dataset
    corpus = base / "corpus.jsonl"
    queries = base / "queries.jsonl"
    qrels = base / "qrels" / f"{split}.tsv"
    return corpus, queries, qrels


def _parse_ks(ks_str: str):
    try:
        return tuple(sorted({int(x) for x in ks_str.split(",") if x.strip()}))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Bad --ks value: {ks_str!r}") from e


def build_parser():
    p = argparse.ArgumentParser("BM25 on BEIR (NFCorpus)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Grid eval
    g = sub.add_parser("grid", help="Evaluate a (k1,b) grid and write metrics + TREC run")
    g.add_argument("--data_root", default="data", help="root data folder")
    g.add_argument("--dataset", default="nfcorpus", choices=["nfcorpus"], help="dataset name")
    g.add_argument("--split", default="test", choices=["train", "dev", "test"], help="qrels split")
    g.add_argument("--ks", type=_parse_ks, default=(1, 5, 10, 100),
                   help="comma-separated cutoffs, e.g., '1,5,10,100'")
    g.add_argument("--k", type=int, default=10, help="(deprecated) kept for compat; ignored if --ks set")
    g.add_argument(
        "--grid",
        default="0.8,0.3;0.8,0.4;0.8,0.5;0.8,0.6;0.8,0.7;0.9,0.3;0.9,0.4;0.9,0.5;0.9,0.6;0.9,0.7;1.0,0.3;1.0,0.4;1.0,0.5;1.0,0.6;1.0,0.7;1.1,0.3;1.1,0.4;1.1,0.5;1.1,0.6;1.1,0.7;1.2,0.3;1.2,0.4;1.2,0.5;1.2,0.6;1.2,0.7;1.3,0.3;1.3,0.4;1.3,0.5;1.3,0.6;1.3,0.7;1.4,0.3;1.4,0.4;1.4,0.5;1.4,0.6;1.4,0.7;1.5,0.3;1.5,0.4;1.5,0.5;1.5,0.6;1.5,0.7;1.6,0.3;1.6,0.4;1.6,0.5;1.6,0.6;1.6,0.7",
        help="semicolon-separated pairs k1,b (e.g. 0.9,0.5;1.2,0.5)",
    )
    g.add_argument("--out_csv", default=None, help="metrics CSV path (defaults under outputs/metrics)")
    g.add_argument("--run_path", default=None, help="TREC run path (defaults under outputs/runs)")

    # Ad-hoc search
    s = sub.add_parser("search", help="Ad-hoc search for a single query")
    s.add_argument("--data_root", default="data")
    s.add_argument("--dataset", default="nfcorpus", choices=["nfcorpus"])
    s.add_argument("--k1", type=float, default=0.9)
    s.add_argument("--b", type=float, default=0.5)
    s.add_argument("--topk", type=int, default=10)
    s.add_argument("--query", required=True, help="your search text")

    return p


def do_grid(args):
    root = Path(args.data_root)
    corpus_p, queries_p, qrels_p = data_paths(root, args.dataset, args.split)

    df_corpus = load_corpus_df(corpus_p)
    queries = load_queries(queries_p)
    gold, info = load_qrels_robust(qrels_p)

    qids = set(gold.keys())
    queries = [(qid, text) for qid, text in queries if qid in qids]

    print(
        f"Loaded dataset={args.dataset} split={args.split} | "
        f"corpus={len(df_corpus):,} queries_with_qrels={len(queries):,}"
    )

    doc_ids, tokens = build_tokens(df_corpus, tokenize_finance)

    grid = []
    for pair in args.grid.split(";"):
        k1s, bs = pair.split(",")
        grid.append((float(k1s), float(bs)))

    df = small_grid(tokens, doc_ids, queries, gold, tokenize_finance, grid, Ks=args.ks)

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else Path("outputs/metrics") / f"{args.dataset}_{args.split}_bm25_grid.csv"
    )
    run_path = (
        Path(args.run_path)
        if args.run_path
        else Path("outputs/runs") / f"{args.dataset}_{args.split}.bm25_sw.trec"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved metrics →", out_csv)

    best = df.iloc[0]
    k1, b = float(best.k1), float(best.b)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run(tokens, doc_ids, queries, tokenize_finance, k1, b, run_path, tag=f"bm25_sw_{args.dataset}")
    print("Wrote TREC run →", run_path)


def do_search(args):
    from rank_bm25 import BM25Okapi
    import numpy as np

    root = Path(args.data_root)
    base = root / args.dataset
    df_corpus = load_corpus_df(base / "corpus.jsonl")
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
        score = float(scores[i])
        text = df_corpus.iloc[i]["content"]
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
