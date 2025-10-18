# save as pca2_from_embeddings.py
# pip install -U numpy scikit-learn torch
import os, numpy as np

EMB = "embeddings.npy" if os.path.exists("embeddings.npy") else "embedding.npy"  # try both
OUT = "embeddings_pca2.npy"

# --- load matrix (supports np.save or torch.save formats)
try:
    X = np.load(EMB)
except Exception:
    import torch
    X = torch.load(EMB)
    if hasattr(X, "detach"): X = X.detach().cpu().numpy()
    elif hasattr(X, "numpy"): X = X.cpu().numpy()

X = np.asarray(X)
assert X.ndim == 2, f"expected 2D array, got shape {X.shape}"

# --- PCA to k=2
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
X2 = pca.fit_transform(X)  # shape: [n_samples, 2]

# --- save & print
np.save(OUT, X2)
print(X2)                 # returns the matrix
print(f"Saved 2D matrix to {OUT} with shape {X2.shape}")