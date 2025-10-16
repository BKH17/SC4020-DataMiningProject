import math

def mrr_at_k(pred_ids, gold_ids, k=10):
    for rank, did in enumerate(pred_ids[:k], 1):
        if did in gold_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(pred_ids, gold_ids, k=10):
    if not gold_ids:
        return 0.0
    return len(set(pred_ids[:k]) & gold_ids) / len(gold_ids)


def ndcg_at_k(pred_ids, gold_ids, k=10):
    dcg = 0.0
    for i, did in enumerate(pred_ids[:k], 1):
        if did in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0
