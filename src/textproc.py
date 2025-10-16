import re, nltk
from nltk.corpus import stopwords

# Ensure stopwords available
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

NEGATIONS = {
    "no", "not", "nor", "without", "isn't", "wasn't", "aren't", "don't", "doesn't",
    "didn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't", "never"
}
STOPS = {w for w in stopwords.words("english") if w not in NEGATIONS}


def tokenize_finance(text: str):
    """Lowercase, keep letters+digits (tickers/numbers), drop stopwords but keep negations."""
    toks = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())
    return [t for t in toks if t not in STOPS]
