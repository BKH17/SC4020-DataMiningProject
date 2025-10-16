from pathlib import Path
import json, gzip, pandas as pd
from collections import defaultdict


def _looks_gzip(p: Path) -> bool:
    with open(p, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def read_jsonl(path: Path):
    opener = gzip.open if _looks_gzip(path) else open
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path.name}: bad JSON at line {i}: {e}") from e
    if not rows:
        raise ValueError(f"{path.name} had no valid JSON lines.")
    return rows


def load_corpus_df(path: Path):
    """Return DataFrame with columns: _id, content (title+text if no content)."""
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)

    if "_id" not in df.columns:
        raise ValueError("corpus must contain '_id'.")

    if "content" in df.columns:
        content = df["content"].fillna("").astype(str)
    else:
        title = df.get("title", "").fillna("").astype(str)
        text = df.get("text", "").fillna("").astype(str)
        content = (title.str.strip() + " " + text).str.strip()

    out = pd.DataFrame({
        "_id": df["_id"].astype(str),
        "content": content.astype(str)
    })
    return out


def load_queries(path: Path):
    """Return list[(qid:str, text:str)] from queries.jsonl."""
    rows = read_jsonl(path)
    out = []
    for o in rows:
        if "_id" in o and "text" in o:
            out.append((str(o["_id"]), str(o["text"])))
    if not out:
        raise ValueError("queries.jsonl missing required fields '_id' and 'text'.")
    return out


def load_qrels_robust(path: Path):
    """Normalize qrels headers across datasets.

    Accepts (case-insensitive) variants:
      qid | query-id | query_id
      docid | corpus-id | corpus_id | doc_id
      score | relevance | label
    """
    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"Empty qrels: {path}")

    cols_l = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols_l:
                return cols_l[n]
        raise ValueError(f"Missing any of {names}. Available: {list(df.columns)}")

    qid_col = pick("qid", "query-id", "query_id")
    doc_col = pick("docid", "corpus-id", "corpus_id", "doc_id")
    score_col = pick("score", "relevance", "label")

    df[qid_col] = df[qid_col].astype(str)
    df[doc_col] = df[doc_col].astype(str)

    gold = defaultdict(set)
    for _, r in df.iterrows():
        val = r[score_col]
        try:
            rel = int(val) if pd.notna(val) else 0
        except Exception:
            rel = int(float(val)) if pd.notna(val) else 0
        if rel > 0:
            gold[r[qid_col]].add(r[doc_col])

    info = {
        "rows": len(df),
        "qid_col": qid_col,
        "doc_col": doc_col,
        "score_col": score_col,
        "unique_qids": len(set(df[qid_col])),
    }
    return gold, info
