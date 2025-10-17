import logging
import os
import pathlib
import random
import pandas as pd

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
dataset = "nfcorpus"
# Use local data directory instead of downloading
project_root = pathlib.Path(__file__).parent.parent.absolute()
data_path = os.path.join(project_root, "data", dataset)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Load bge model using Sentence Transformers
model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=128)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

# Retrieve dense results
results = retriever.retrieve(corpus, queries)

# Evaluate retrieval
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# Save TREC run file
run_path = pathlib.Path("outputs/runs") / f"{dataset}_test.bge_base.trec"
run_path.parent.mkdir(parents=True, exist_ok=True)
with open(run_path, "w") as f:
    for query_id, doc_scores in results.items():
        # Sort by score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\tbge_base_{dataset}\n")
logging.info(f"Saved TREC run file to: {run_path}")

# Save metrics to CSV
metrics_data = []
for k in retriever.k_values:
    metrics_data.append({
        "model": "bge-base-en-v1.5",
        "dataset": dataset,
        "split": "test",
        "k": k,
        "NDCG@k": ndcg.get(f"NDCG@{k}", 0.0),
        "MAP@k": _map.get(f"MAP@{k}", 0.0),
        "Recall@k": recall.get(f"Recall@{k}", 0.0),
        "P@k": precision.get(f"P@{k}", 0.0)
    })

df_metrics = pd.DataFrame(metrics_data)
metrics_path = pathlib.Path("outputs/metrics") / f"{dataset}_test_bge_base.csv"
metrics_path.parent.mkdir(parents=True, exist_ok=True)
df_metrics.to_csv(metrics_path, index=False)
logging.info(f"Saved metrics to: {metrics_path}")
logging.info(f"\n{df_metrics.to_string(index=False)}")