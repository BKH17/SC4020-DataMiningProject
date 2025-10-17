import logging
import os
import pathlib
import pandas as pd

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from rank_bm25 import BM25Okapi

# Handle both direct execution and module import
try:
    from .textproc import tokenize_finance
except ImportError:
    from textproc import tokenize_finance

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Configuration
dataset = "nfcorpus"
project_root = pathlib.Path(__file__).parent.parent.absolute()
data_path = os.path.join(project_root, "data", dataset)

# BM25 parameters
K1 = 0.9
B = 0.5

# Alpha values to test (0.0 to 1.0 in steps of 0.1)
ALPHA_VALUES = [round(x * 0.1, 1) for x in range(11)]  # [0.0, 0.1, ..., 1.0]

# Evaluation cutoffs
K_VALUES = [10]


def normalize_scores(scores_dict):
    """
    Normalize scores to [0, 1] range using max normalization.

    Args:
        scores_dict: Dict[query_id, Dict[doc_id, score]]

    Returns:
        Dict[query_id, Dict[doc_id, normalized_score]]
    """
    normalized = {}
    for query_id, doc_scores in scores_dict.items():
        if not doc_scores:
            normalized[query_id] = {}
            continue

        max_score = max(doc_scores.values())
        if max_score > 0:
            normalized[query_id] = {
                doc_id: score / max_score
                for doc_id, score in doc_scores.items()
            }
        else:
            normalized[query_id] = doc_scores

    return normalized


def compute_bm25_scores(corpus, queries, k1=0.9, b=0.5):
    """
    Compute BM25 scores for all queries.

    Args:
        corpus: Dict[doc_id, {'title': str, 'text': str}]
        queries: Dict[query_id, str]
        k1: BM25 k1 parameter
        b: BM25 b parameter

    Returns:
        Dict[query_id, Dict[doc_id, score]]
    """
    logging.info(f"Computing BM25 scores with k1={k1}, b={b}...")

    # Prepare corpus documents
    doc_ids = list(corpus.keys())
    doc_texts = []
    for doc_id in doc_ids:
        doc = corpus[doc_id]
        # Combine title and text like in utils_io.py
        title = doc.get('title', '').strip()
        text = doc.get('text', '').strip()
        content = f"{title} {text}".strip()
        doc_texts.append(content)

    # Tokenize corpus
    logging.info("Tokenizing corpus...")
    tokenized_corpus = [tokenize_finance(doc) for doc in doc_texts]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Compute scores for each query
    results = {}
    for query_id, query_text in queries.items():
        tokenized_query = tokenize_finance(query_text)
        scores = bm25.get_scores(tokenized_query)

        # Create dict of doc_id -> score
        results[query_id] = {
            doc_ids[i]: float(scores[i])
            for i in range(len(doc_ids))
        }

    logging.info(f"BM25 scoring complete for {len(queries)} queries")
    return results


def compute_hybrid_scores(bm25_scores, dense_scores, alpha):
    """
    Combine BM25 and dense scores using hybrid formula.

    hybrid_score = (1 - alpha) * bm25_score + alpha * dense_score

    Args:
        bm25_scores: Dict[query_id, Dict[doc_id, normalized_score]]
        dense_scores: Dict[query_id, Dict[doc_id, normalized_score]]
        alpha: Weight parameter [0, 1]

    Returns:
        Dict[query_id, Dict[doc_id, hybrid_score]]
    """
    hybrid_results = {}

    for query_id in bm25_scores.keys():
        bm25_query = bm25_scores.get(query_id, {})
        dense_query = dense_scores.get(query_id, {})

        # Get all doc_ids from both methods
        all_doc_ids = set(bm25_query.keys()) | set(dense_query.keys())

        hybrid_results[query_id] = {}
        for doc_id in all_doc_ids:
            bm25_score = bm25_query.get(doc_id, 0.0)
            dense_score = dense_query.get(doc_id, 0.0)

            # Hybrid formula
            hybrid_score = (1 - alpha) * bm25_score + alpha * dense_score
            hybrid_results[query_id][doc_id] = hybrid_score

    return hybrid_results


def save_trec_run(results, output_path, run_tag):
    """
    Save results in TREC format.

    Args:
        results: Dict[query_id, Dict[doc_id, score]]
        output_path: Path to save TREC file
        run_tag: Tag to identify the run
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for query_id, doc_scores in results.items():
            # Sort by score descending
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_tag}\n")


def main():
    logging.info("="*80)
    logging.info("HYBRID RETRIEVAL EVALUATION: BM25 + BGE")
    logging.info("="*80)

    # Load dataset
    logging.info(f"Loading {dataset} test set...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    logging.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")

    # 1. Compute BM25 scores
    bm25_results = compute_bm25_scores(corpus, queries, k1=K1, b=B)
    bm25_normalized = normalize_scores(bm25_results)
    logging.info("BM25 scores computed and normalized")

    # 2. Compute BGE dense scores
    logging.info("Loading BGE model (BAAI/bge-base-en-v1.5)...")
    model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=K_VALUES)

    logging.info("Computing dense retrieval scores...")
    dense_results = retriever.retrieve(corpus, queries)
    dense_normalized = normalize_scores(dense_results)
    logging.info("Dense scores computed and normalized")

    # 3. Evaluate for different alpha values
    logging.info("="*80)
    logging.info(f"Testing {len(ALPHA_VALUES)} alpha values: {ALPHA_VALUES}")
    logging.info("="*80)

    all_metrics = []

    for alpha in ALPHA_VALUES:
        logging.info(f"\n>>> Evaluating alpha={alpha:.1f} <<<")

        # Compute hybrid scores
        hybrid_results = compute_hybrid_scores(bm25_normalized, dense_normalized, alpha)

        # Evaluate
        ndcg, _map, recall, precision = retriever.evaluate(qrels, hybrid_results, K_VALUES)

        # Store metrics
        for k in K_VALUES:
            all_metrics.append({
                "model": f"hybrid_alpha_{alpha:.1f}",
                "dataset": dataset,
                "split": "test",
                "alpha": alpha,
                "k1": K1,
                "b": B,
                "k": k,
                "NDCG@k": ndcg.get(f"NDCG@{k}", 0.0),
                "MAP@k": _map.get(f"MAP@{k}", 0.0),
                "Recall@k": recall.get(f"Recall@{k}", 0.0),
                "P@k": precision.get(f"P@{k}", 0.0)
            })

        # Log summary for this alpha
        logging.info(f"  NDCG@10: {ndcg.get('NDCG@10', 0.0):.4f}")
        logging.info(f"  MAP@10:  {_map.get('MAP@10', 0.0):.4f}")
        logging.info(f"  Recall@10: {recall.get('Recall@10', 0.0):.4f}")

        # Save TREC run file for this alpha
        run_path = pathlib.Path("outputs/runs") / f"{dataset}_test.hybrid_alpha_{alpha:.1f}.trec"
        save_trec_run(hybrid_results, run_path, f"hybrid_alpha_{alpha:.1f}")
        logging.info(f"  Saved TREC run: {run_path}")

    # 4. Save all metrics to CSV
    df_metrics = pd.DataFrame(all_metrics)
    metrics_path = pathlib.Path("outputs/metrics") / f"{dataset}_test_hybrid.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(metrics_path, index=False)

    logging.info("="*80)
    logging.info(f"All metrics saved to: {metrics_path}")
    logging.info("="*80)

    # 5. Display results table for all alpha values @ k=10
    logging.info("\nResults for all alpha values @ k=10:")
    logging.info("="*80)
    results_table = df_metrics[['alpha', 'NDCG@k', 'MAP@k', 'Recall@k', 'P@k']].sort_values('alpha')
    print(results_table.to_string(index=False))

    logging.info("\n" + "="*80)
    logging.info("HYBRID EVALUATION COMPLETE")
    logging.info("="*80)


if __name__ == "__main__":
    main()
