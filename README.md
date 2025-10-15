# SC4020 – BM25 Similarity Search on FiQA-2018

A minimal, reproducible pipeline for Okapi BM25 retrieval on the FiQA-2018 dataset. It supports:
- Stop-word–aware tokenization (keeps negations like not, never)
- Hyperparameter sweep for k1 and b with MRR@k / Recall@k / nDCG@k
- TREC run file export
- Ad-hoc top-k search for any query

### Project structure

```text
SC4020-DataMiningProject/
├─ data/
│  ├─ corpus.jsonl
│  ├─ queries.jsonl
│  └─ fiqa_qrels_test.tsv
├─ outputs/
│  ├─ metrics/
│  └─ runs/
├─ src/
│  ├─ cli.py
│  ├─ bm25_runner.py
│  ├─ metrics.py
│  ├─ textproc.py
│  └─ utils_io.py
└─ requirements.txt
```

### Quick Start
1. **Clone the repository**
```bash
git clone https://github.com/BKH17/SC4020-DataMiningProject.git
cd SC4020-DataMiningProject
```

2. **Create a virtual environment & install**

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. **Data files**

Put these files under `data/`:
- `data/corpus.jsonl`          # FiQA corpus (57,638 docs)
- `data/queries.jsonl`         # 6,648 queries
- `data/fiqa_qrels_test.tsv`   # qrels (648 queries with positives)

### How to run

A) ****Hyperparameter sweep + TREC run****

Runs BM25 for a small grid of (k1,b), evaluates against qrels, writes a metrics CSV and a TREC run file.
```bash
mkdir -p outputs/metrics outputs/runs

python3 -m src.cli grid \
  --corpus data/corpus.jsonl \
  --queries data/queries.jsonl \
  --qrels data/fiqa_qrels_test.tsv \
  --k 10 \
  --grid "0.9,0.5;1.2,0.5;1.5,0.5;0.9,0.75;1.2,0.75;1.5,0.75" \
  --out_csv outputs/metrics/bm25_small_grid.csv \
  --run_path outputs/runs/run.bm25_sw.trec
```

Outputs:
- outputs/metrics/bm25_small_grid.csv – one row per (k1,b) with MRR@10, Recall@10, nDCG@10, Seconds
- outputs/runs/run.bm25_sw.trec – TREC run file:
```
<qid> Q0 <docid> <rank> <score> bm25_sw
```

B) **Ad-hoc search (no qrels needed)**

Returns top-k docs for a free-text query with chosen BM25 params.
```bash
python3 -m src.cli search \
  --corpus data/corpus.jsonl \
  --k1 0.9 --b 0.5 --topk 5 \
  --query "difference between stock split and reverse stock split"
```
