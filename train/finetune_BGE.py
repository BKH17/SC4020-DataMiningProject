import os, time, json, random, logging
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
import tqdm

# --------------------
# Config (edit here)
# --------------------
DATA_DIR   = "data/nfcorpus"
TRAIN_SPLIT= "train"
EVAL_SPLIT = "test"      # set "" to skip eval
MODEL_NAME = "BAAI/bge-base-en-v1.5"
OUTPUT_DIR = "outputs/bge-base-en-v15_nfcorpus"

EPOCHS       = 5
BATCH_SIZE   = 256
EVAL_BATCH   = 128
LR           = 2e-5
WEIGHT_DECAY = 0.0
WARMUP_FRAC  = 0.1
GRAD_ACCUM   = 1
MAX_Q_LEN    = 128
MAX_D_LEN    = 256
EMA_BETA     = 0.98
SAVE_EVERY   = 2000      # optimizer steps (0 = disable)
LOG_EVERY    = 200
USE_AMP      = True
SEED         = 42
K_VALUES     = [1, 3, 5, 10, 100]

# --------------------
# Helpers
# --------------------
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def bge_query(q): 
    return f"Represent this sentence for searching relevant passages: {q}"

def join_doc(d):
    t = (d.get("title") or "").strip()
    x = (d.get("text") or "").strip()
    return (t + " \n" + x).strip() if t else x

def build_pairs(corpus, queries, qrels, max_q_chars, max_d_chars):
    ex = []
    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        q = bge_query(queries[qid])[:max_q_chars]
        for pid, s in rels.items():
            if s > 0 and pid in corpus:
                p = join_doc(corpus[pid])[:max_d_chars]
                if p:
                    ex.append(InputExample(texts=[q, p]))
    random.shuffle(ex)
    return ex

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def to_device(feats, device):
    """Robustly move features to device across sentence-transformers versions."""
    if isinstance(feats, tuple):   # some versions return (features, labels)
        feats = feats[0]
    if isinstance(feats, list):
        return [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in f.items()} for f in feats]
    if isinstance(feats, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in feats.items()}
    return feats

def mnrl_loss(loss_fct, feats, batch_len):
    """Compatibility shim: ST v2 requires labels Tensor; ST v3+ accepts labels=None."""
    try:
        return loss_fct(feats, labels=None)
    except TypeError:
        device = feats[0]["input_ids"].device if isinstance(feats, list) else next(iter(feats.values())).device
        dummy = torch.zeros(batch_len, dtype=torch.long, device=device)
        return loss_fct(feats, labels=dummy)

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "ckpts"), exist_ok=True)

    # Save run config
    save_json(os.path.join(OUTPUT_DIR, "config.json"), {
        "data_dir": DATA_DIR, "train_split": TRAIN_SPLIT, "eval_split": EVAL_SPLIT,
        "model_name": MODEL_NAME, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "warmup_frac": WARMUP_FRAC, "grad_accum": GRAD_ACCUM,
        "max_q_len": MAX_Q_LEN, "max_d_len": MAX_D_LEN, "ema_beta": EMA_BETA,
        "save_every": SAVE_EVERY, "log_every": LOG_EVERY, "use_amp": USE_AMP, "seed": SEED
    })

    # Load BEIR data
    corpus_tr, queries_tr, qrels_tr = GenericDataLoader(DATA_DIR).load(split=TRAIN_SPLIT)
    logging.info(f"Train sizes | corpus={len(corpus_tr):,} queries={len(queries_tr):,} qrels={len(qrels_tr):,}")

    # Build training pairs
    train_examples = build_pairs(corpus_tr, queries_tr, qrels_tr, MAX_Q_LEN*4, MAX_D_LEN*4)
    assert len(train_examples) > 0, "No training pairs—check data path/qrels."

    # Model
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = max(MAX_Q_LEN, MAX_D_LEN)

    # Keep InputExample objects raw; avoid default collate
    train_loader = DataLoader(
        train_examples,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: x,
    )

    loss_fct = losses.MultipleNegativesRankingLoss(model)

    total_steps = (len(train_loader) * EPOCHS) // max(1, GRAD_ACCUM)
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = LinearLR(optimizer, start_factor=1.0 / max(1, warmup_steps), total_iters=warmup_steps)
    scaler = GradScaler(enabled=USE_AMP)

    # Loss logging
    loss_csv = os.path.join(OUTPUT_DIR, "loss.csv")
    with open(loss_csv, "w", encoding="utf-8") as f:
        f.write("step,epoch,loss,ema,lr,elapsed_sec\n")
    ema, start_time, step = None, time.time(), 0

    # Train
    model.train()
    for epoch in range(1, EPOCHS + 1):
        for it, batch in tqdm.tqdm(enumerate(train_loader, start=1), total=len(train_loader)):
            feats = model.smart_batching_collate(batch)
            feats = to_device(feats, model.device)

            with autocast(enabled=USE_AMP):
                loss = mnrl_loss(loss_fct, feats, batch_len=len(batch))

            scaler.scale(loss / GRAD_ACCUM).backward()

            if it % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if step < warmup_steps:
                    scheduler.step()
                step += 1

                cur = float(loss.detach().item())
                ema = cur if ema is None else (EMA_BETA * ema + (1 - EMA_BETA) * cur)
                ema_corr = ema / (1 - (EMA_BETA ** step))
                elapsed = time.time() - start_time
                lr_now = optimizer.param_groups[0]["lr"]
                with open(loss_csv, "a", encoding="utf-8") as f:
                    f.write(f"{step},{epoch},{cur:.6f},{ema_corr:.6f},{lr_now:.8f},{elapsed:.2f}\n")

                if step % LOG_EVERY == 0:
                    logging.info(f"step {step}/{total_steps} | epoch {epoch} | loss={cur:.4f} ema={ema_corr:.4f} lr={lr_now:.2e}")

                if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                    ck = os.path.join(OUTPUT_DIR, "ckpts", f"step-{step:06d}")
                    os.makedirs(ck, exist_ok=True)
                    model.save(ck)
                    logging.info(f"Saved checkpoint -> {ck}")

    # Save final model
    model.save(OUTPUT_DIR)
    logging.info(f"Training complete. Saved to: {OUTPUT_DIR}")

    # Eval on dev split (skip if EVAL_SPLIT == "")
    if EVAL_SPLIT:
        logging.info(f"Evaluating on split: {EVAL_SPLIT}")
        corpus_ev, queries_ev, qrels_ev = GenericDataLoader(DATA_DIR).load(split=EVAL_SPLIT)
        beir_model = DRES(models.SentenceBERT(OUTPUT_DIR), batch_size=EVAL_BATCH)
        retriever = EvaluateRetrieval(beir_model, score_function="cos_sim")
        results = retriever.retrieve(corpus_ev, queries_ev)
        ndcg, _map, recall, precision = retriever.evaluate(qrels_ev, results, k_values=K_VALUES)
        metrics = {
            "split": EVAL_SPLIT,
            "k_values": K_VALUES,
            "nDCG": {str(k): float(ndcg.get(k, float("nan"))) for k in K_VALUES},
            "MAP": {str(k): float(_map.get(k, float("nan"))) for k in K_VALUES},
            "Recall": {str(k): float(recall.get(k, float("nan"))) for k in K_VALUES},
            "Precision": {str(k): float(precision.get(k, float("nan"))) for k in K_VALUES},
        }
        save_json(os.path.join(OUTPUT_DIR, "metrics.json"), metrics)
        logging.info("Eval metrics saved -> metrics.json")
