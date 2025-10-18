# save as make_bundle_with_labels.py
import json

TEST, CORPUS, QUERY, OUT = 'TEST_FILE', "/nfcorpus/corpus.jsonl", "nfcorpus/queries.jsonl", "bundle.json"

# --- 1) read first qid block + collect 3 unrelated doc ids from following qids
first_qid, rel_docs, unrelated = None, [], []
with open(TEST, encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln: continue
        a = ln.split("\t")
        if a[0].lower() in {"qid","query-id","query_id"}:  # skip header
            continue
        qid, pid, *rest = a + ["0"]*(3-len(a))  # tolerate 2 or 3 cols
        score = int(rest[0])
        if first_qid is None:
            first_qid = qid
        if qid == first_qid:
            rel_docs.append((pid, score))
        elif len(unrelated) < 3:
            unrelated.append(pid)
        if first_qid and len(unrelated) == 3 and qid != first_qid and rel_docs:
            # we have everything needed
            pass

# dedupe unrelated (avoid overlap with related)
rel_pids = {pid for pid, _ in rel_docs}
unrelated = [pid for pid in unrelated if pid not in rel_pids][:3]

# --- 2) look up query text
qid2text = {}
with open(QUERY, encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        o = json.loads(ln)
        qid2text[o["_id"]] = o.get("text", "")
query_text = qid2text.get(first_qid, "")

# --- 3) fetch needed docs from corpus
need = rel_pids | set(unrelated)
pid2text = {}
with open(CORPUS, encoding="utf-8") as f:
    for ln in f:
        if len(pid2text) == len(need): break
        if not ln.strip(): continue
        o = json.loads(ln)
        pid = o.get("_id")
        if pid in need:
            title = (o.get("title") or "").strip()
            text  = (o.get("text")  or "").strip()
            pid2text[pid] = (title+"\n"+text).strip() if title else text

# --- 4) build labeled docs (+ add unrelated as red)
def label(score): return "green" if score==2 else ("yellow" if score==1 else "red")

docs = [{"doc_id": pid, "text": pid2text.get(pid, ""), "label": label(sc)} for pid, sc in rel_docs]
docs += [{"doc_id": pid, "text": pid2text.get(pid, ""), "label": "red"} for pid in unrelated]

bundle = {"query_id": first_qid, "query": query_text, "documents": docs}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(bundle, f, ensure_ascii=False, indent=2)

print(f"Saved {OUT} for query {first_qid} with {len(docs)} docs "
      f"({len(rel_docs)} related + {len(unrelated)} unrelated).")