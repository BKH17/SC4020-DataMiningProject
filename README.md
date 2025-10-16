# SC4020 – BM25 Similarity Search on BEIR (FiQA-2018 & NFCorpus)

A minimal, reproducible pipeline for Okapi **BM25** retrieval on two BEIR datasets:
- **FiQA-2018** (finance QA)
- **NFCorpus** (consumer health)

Features:
- Stop-word–aware tokenization (keeps negations like *not*, *never*)
- Hyperparameter sweep for **k1** and **b** with **MRR@k / Recall@k / nDCG@k**
- TREC run file export
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

### A) ****Hyperparameter sweep (tune on dev) + TREC run****

Runs a grid of (k1,b), evaluates against qrels, writes a metrics CSV and a TREC run file.

Make directory:
```bash
mkdir -p outputs/metrics outputs/runs
```
FiQA:
```bash
python3 -m src.cli grid --dataset fiqa --split dev \
  --grid "0.8,0.4;0.8,0.5;0.8,0.6;0.8,0.7;0.9,0.4;0.9,0.5;0.9,0.6;0.9,0.7;1.0,0.4;1.0,0.5;1.0,0.6;1.0,0.7;1.2,0.4;1.2,0.5;1.2,0.6;1.2,0.7;1.4,0.4;1.4,0.5;1.4,0.6;1.4,0.7;1.6,0.4;1.6,0.5;1.6,0.6;1.6,0.7"
```
NFCorpus:
```bash
python3 -m src.cli grid --dataset nfcorpus --split dev \
  --grid "0.8,0.4;0.8,0.5;0.8,0.6;0.8,0.7;0.9,0.4;0.9,0.5;0.9,0.6;0.9,0.7;1.0,0.4;1.0,0.5;1.0,0.6;1.0,0.7;1.2,0.4;1.2,0.5;1.2,0.6;1.2,0.7;1.4,0.4;1.4,0.5;1.4,0.6;1.4,0.7;1.6,0.4;1.6,0.5;1.6,0.6;1.6,0.7"
```


Outputs:
- csv files contain metrics: one row per (k1,b) with MRR@10, Recall@10, nDCG@10, Seconds
- TREC run files that contain:
```
<qid> Q0 <docid> <rank> <score> bm25_sw
```

### B) **Ad-hoc search (no qrels needed)**

Returns top-k docs for a free-text query with chosen BM25 params.
```bash
python3 -m src.cli search --dataset fiqa \
  --k1 0.8 --b 0.4 --topk 5 \
  --query "difference between stock split and reverse stock split"
  
python3 -m src.cli search --dataset nfcorpus \
  --k1 1.4 --b 0.4 --topk 5 \
  --query "phosphorus and cardiovascular risk"
```
