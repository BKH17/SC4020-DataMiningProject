# SC4020 – Information Retrieval on BEIR (FiQA-2018 & NFCorpus)

A minimal, reproducible pipeline for **BM25** and **BGE** (dense) retrieval on two BEIR datasets:
- **FiQA-2018** (finance QA)
- **NFCorpus** (consumer health)

Features:
- **BM25**: Stop-word–aware tokenization (keeps negations like *not*, *never*)
- **BM25**: Hyperparameter sweep for **k1** and **b** with **MRR@k / Recall@k / nDCG@k**
- **BGE**: Dense retrieval using BAAI/bge-base-en-v1.5 sentence embeddings
- TREC run file export for both methods
- Ad-hoc top-k search for any query
- Dataset/split-aware CLI (`--dataset {fiqa|nfcorpus}`, `--split {train|dev|test}`)

## Project structure

```text
SC4020-DataMiningProject/
├─ data/
│ ├─ fiqa/
│ │ ├─ corpus.jsonl
│ │ ├─ queries.jsonl
│ │ └─ qrels/
│ │ ├─ train.tsv
│ │ ├─ dev.tsv
│ │ └─ test.tsv
│ └─ nfcorpus/
│ ├─ corpus.jsonl
│ ├─ queries.jsonl
│ └─ qrels/
│ ├─ train.tsv
│ ├─ dev.tsv
│ └─ test.tsv
├─ outputs/
│ ├─ metrics/
│ └─ runs/
├─ src/
│ ├─ cli.py
│ ├─ bm25_runner.py
│ ├─ bge_evaluation.py
│ ├─ metrics.py
│ ├─ textproc.py
│ └─ utils_io.py
└─ requirements.txt
```

**Notes on file formats**
- `corpus.jsonl`: objects with `_id`, and either `content` **or** (`title`, `text`) — we auto-compose `content = title + text`.
- `queries.jsonl`: objects with `_id`, `text`.
- `qrels/*.tsv`: any of the following headers (case-insensitive) are accepted:  
  - query id: `qid | query-id | query_id`  
  - doc id: `docid | corpus-id | corpus_id | doc_id`  
  - relevance: `score | relevance | label` (we treat **rel > 0** as relevant ⇒ binary-gain metrics).

---

## Quick Start
### 1) **Clone the repository**
```bash
git clone https://github.com/BKH17/SC4020-DataMiningProject.git
cd SC4020-DataMiningProject
```

### 2. **Create venv & install**

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
python -m pip install -U pip
python -m pip install -r requirements.txt
```

---

## How to run

### A) **BM25: Hyperparameter sweep (tune on dev) + TREC run**

Runs a grid of (k1,b), evaluates against qrels, writes a metrics CSV and a TREC run file.

Make directory:
```bash
mkdir -p outputs/metrics outputs/runs
```
FiQA:
```bash
python3 -m src.cli grid --dataset fiqa --split dev \
  --grid "0.8,0.3;0.8,0.4;0.8,0.5;0.8,0.6;0.8,0.7;0.9,0.3;0.9,0.4;0.9,0.5;0.9,0.6;0.9,0.7;1.0,0.3;1.0,0.4;1.0,0.5;1.0,0.6;1.0,0.7;1.1,0.3;1.1,0.4;1.1,0.5;1.1,0.6
;1.1,0.7;1.2,0.3;1.2,0.4;1.2,0.5;1.2,0.6;1.2,0.7;1.3,0.3;1.3,0.4;1.3,0.5;1.3,0.6;1.3,0.7;1.4,0.3;1.4,0.4;1.4,0.5;1.4,0.6;1.4,0.7;1.5,0.3;1.5,0.4;1.5,0.5;1.5,0.6;1.5,0.7;1.6,0.3;1.6,0.4;1.6,0.5;1.6,0.6;1.6,0.7"
```
NFCorpus:
```bash
python3 -m src.cli grid --dataset nfcorpus --split dev \
  --grid "0.8,0.3;0.8,0.4;0.8,0.5;0.8,0.6;0.8,0.7;0.9,0.3;0.9,0.4;0.9,0.5;0.9,0.6;0.9,0.7;1.0,0.3;1.0,0.4;1.0,0.5;1.0,0.6;1.0,0.7;1.1,0.3;1.1,0.4;1.1,0.5;1.1,0.6
;1.1,0.7;1.2,0.3;1.2,0.4;1.2,0.5;1.2,0.6;1.2,0.7;1.3,0.3;1.3,0.4;1.3,0.5;1.3,0.6;1.3,0.7;1.4,0.3;1.4,0.4;1.4,0.5;1.4,0.6;1.4,0.7;1.5,0.3;1.5,0.4;1.5,0.5;1.5,0.6;1.5,0.7;1.6,0.3;1.6,0.4;1.6,0.5;1.6,0.6;1.6,0.7"
```


Outputs:
- **Metrics CSVs** (one per run): each row is a (k1, b) setting with results at multiple cutoffs. Columns:
```
N,Seconds,k1,b,
MRR@1,Recall@1,nDCG@1,
MRR@5,Recall@5,nDCG@5,
MRR@10,Recall@10,nDCG@10,
MRR@100,Recall@100,nDCG@100
```
- **TREC run files** (for the best k1,b on the chosen split): one line per (query, doc) with BM25 score.
```
<qid> Q0 <docid> <rank> <score> bm25_sw_<dataset>
```

### B) **BM25: Ad-hoc search (no qrels needed)**

Returns top-k docs for a free-text query with chosen BM25 params.
```bash
python3 -m src.cli search --dataset fiqa \
  --k1 0.8 --b 0.4 --topk 5 \
  --query "difference between stock split and reverse stock split"
  
python3 -m src.cli search --dataset nfcorpus \
  --k1 1.6 --b 0.7 --topk 5 \
  --query "phosphorus and cardiovascular risk"
```

### C) **BGE: Dense retrieval evaluation**

Runs BGE (BAAI/bge-base-en-v1.5) dense retrieval, evaluates on test split, and saves metrics + TREC run file.

**Note:** First run will download the 438MB BGE model (~5-10 minutes depending on connection).

```bash
python3 src/bge_evaluation.py
```

This will:
- Load corpus, queries, and qrels from `data/nfcorpus/`
- Encode documents and queries using BGE embeddings
- Retrieve top documents using cosine similarity
- Evaluate at k=[1, 3, 5, 10, 100, 1000]
- Save metrics to `outputs/metrics/nfcorpus_test_bge_base.csv`
- Save TREC run file to `outputs/runs/nfcorpus_test.bge_base.trec`

**Outputs:**
- Metrics CSV contains: NDCG@k, MAP@k, Recall@k, P@k for all k values
- TREC run file format: `<qid> Q0 <docid> <rank> <score> bge_base_nfcorpus`
