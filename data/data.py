"""
data/data.py
------------
Trains an MLP on MNIST, extracts hidden-layer activations at multiple
epochs (T1) and multiple layers (T2), projects them to 2D via t-SNE,
Procrustes-aligns the embeddings, and writes two JSON data files.

Run from the project root:  python data/data.py

Output:
  data/epochs.json  – T1: inter-epoch evolution (layer-4 activations)
  data/layers.json  – T2: inter-layer evolution (final-epoch activations)
MNIST raw files are downloaded to ./mnist/ (project root, not inside data/).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
EXTRACT_EPOCHS  = [0, 1, 2, 5, 10, 20]
MAX_EPOCH       = max(EXTRACT_EPOCHS)
N_PER_CLASS     = 100          # 100 × 10 classes = 1 000 test observations
TSNE_PERPLEXITY = 30
TSNE_ITER       = 1000          # max_iter (sklearn ≥1.5)
PCA_DIM         = 50           # pre-reduce before t-SNE for speed

# ── Model ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1    = nn.Linear(784, 512)
        self.fc2    = nn.Linear(512, 256)
        self.fc3    = nn.Linear(256, 128)
        self.fc4    = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 10)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x  = x.view(-1, 784)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        out = self.fc_out(h4)
        return out, [h1, h2, h3, h4]

# ── Data helpers ───────────────────────────────────────────────────────────────
def load_mnist():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./mnist', train=True,  download=True, transform=tf)
    test  = datasets.MNIST('./mnist', train=False, download=True, transform=tf)
    return train, test

def balanced_indices(dataset, n_per_class=100):
    indices, counts = [], {i: 0 for i in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if counts[label] < n_per_class:
            indices.append(idx)
            counts[label] += 1
        if all(v >= n_per_class for v in counts.values()):
            break
    return indices

# ── Activation extraction ──────────────────────────────────────────────────────
def extract(model, loader, device):
    model.eval()
    acts   = [[] for _ in range(4)]
    labels = []
    with torch.no_grad():
        for data, lbl in loader:
            _, hiddens = model(data.to(device))
            for i, h in enumerate(hiddens):
                acts[i].append(h.cpu().numpy())
            labels.extend(lbl.numpy())
    return [np.vstack(a) for a in acts], np.array(labels)

def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, lbl in loader:
            out, _ = model(data.to(device))
            pred   = out.argmax(1).cpu()
            correct += (pred == lbl).sum().item()
            total   += len(lbl)
    return round(correct / total * 100, 2)

# ── t-SNE + alignment ──────────────────────────────────────────────────────────
def run_tsne(acts):
    """PCA pre-reduction then t-SNE."""
    dim = min(PCA_DIM, acts.shape[1], acts.shape[0] - 1)
    if dim < acts.shape[1]:
        acts = PCA(n_components=dim, random_state=42).fit_transform(acts)
    emb = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITER,
        init='pca',
        learning_rate='auto',
        random_state=42
    ).fit_transform(acts)
    return emb

def normalize(emb):
    """Scale to [-1, 1] centred at origin."""
    emb = emb - emb.mean(0)
    scale = np.abs(emb).max() + 1e-8
    return emb / scale

def procrustes_align(reference, target):
    """Rotate/reflect *target* to best match *reference* (Procrustes)."""
    ref_n = reference / (np.linalg.norm(reference) + 1e-8)
    tgt_n = target    / (np.linalg.norm(target)    + 1e-8)
    R, _  = orthogonal_procrustes(tgt_n, ref_n)
    aligned = tgt_n @ R * np.linalg.norm(reference)
    return aligned

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs('./data', exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_ds, test_ds = load_mnist()
    test_idx    = balanced_indices(test_ds, N_PER_CLASS)
    test_subset = Subset(test_ds, test_idx)
    test_loader = DataLoader(test_subset, batch_size=1000, shuffle=False)
    train_loader= DataLoader(train_ds,   batch_size=256,  shuffle=True)

    # ── Model + optimizer ──────────────────────────────────────────────────────
    model     = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop with activation extraction ───────────────────────────────
    epoch_acts  = {}   # {epoch: [layer_acts × 4]}
    epoch_stats = {}   # {epoch: {loss, acc}}
    labels      = None

    # Epoch 0 – before any training
    acts, labels = extract(model, test_loader, device)
    epoch_acts[0]  = acts
    epoch_stats[0] = {'loss': None, 'acc': accuracy(model, test_loader, device)}
    print(f"Epoch  0 | acc={epoch_stats[0]['acc']:.1f}%")

    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out, _ = model(data)
            loss   = criterion(out, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        if epoch in EXTRACT_EPOCHS:
            acts, _ = extract(model, test_loader, device)
            acc     = accuracy(model, test_loader, device)
            epoch_acts[epoch]  = acts
            epoch_stats[epoch] = {'loss': round(avg_loss, 4), 'acc': acc}
            print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | acc={acc:.1f}%")

    print("\nTraining complete. Running t-SNE …")

    # ── T1: inter-epoch (layer 4 activations) ─────────────────────────────────
    t1_embs   = {}
    ref_emb   = None
    for ep in EXTRACT_EPOCHS:
        print(f"  t-SNE epoch {ep} …", flush=True)
        emb = run_tsne(epoch_acts[ep][3])      # layer 4 (index 3)
        emb = normalize(emb)
        if ref_emb is None:
            ref_emb = emb
        else:
            emb = normalize(procrustes_align(ref_emb, emb))
        t1_embs[ep] = emb

    t1_points = [
        {
            "id":    int(i),
            "label": int(labels[i]),
            "positions": [
                {"epoch": ep,
                 "x": float(t1_embs[ep][i, 0]),
                 "y": float(t1_embs[ep][i, 1])}
                for ep in EXTRACT_EPOCHS
            ]
        }
        for i in range(len(labels))
    ]

    t1_data = {
        "epochs":    EXTRACT_EPOCHS,
        "n_samples": len(labels),
        "layer":     "Layer 4 – 64 units (closest to output)",
        "stats":     {str(ep): epoch_stats[ep] for ep in EXTRACT_EPOCHS},
        "points":    t1_points
    }

    with open('./data/epochs.json', 'w') as f:
        json.dump(t1_data, f)
    print("Saved data/epochs.json")

    # ── T2: inter-layer (final epoch activations) ──────────────────────────────
    layer_names = [
        "Layer 1 – 512 units",
        "Layer 2 – 256 units",
        "Layer 3 – 128 units",
        "Layer 4 – 64 units"
    ]
    final_acts = epoch_acts[MAX_EPOCH]
    t2_embs    = {}
    ref_emb    = None

    for li in range(4):
        print(f"  t-SNE layer {li+1} …", flush=True)
        emb = run_tsne(final_acts[li])
        emb = normalize(emb)
        if ref_emb is None:
            ref_emb = emb
        else:
            emb = normalize(procrustes_align(ref_emb, emb))
        t2_embs[li] = emb

    t2_points = [
        {
            "id":    int(i),
            "label": int(labels[i]),
            "positions": [
                {"layer": li,
                 "x": float(t2_embs[li][i, 0]),
                 "y": float(t2_embs[li][i, 1])}
                for li in range(4)
            ]
        }
        for i in range(len(labels))
    ]

    t2_data = {
        "layers":    layer_names,
        "n_samples": len(labels),
        "epoch":     MAX_EPOCH,
        "points":    t2_points
    }

    with open('./data/layers.json', 'w') as f:
        json.dump(t2_data, f)
    print("Saved data/layers.json")
    print("\nDone!")

if __name__ == '__main__':
    main()
