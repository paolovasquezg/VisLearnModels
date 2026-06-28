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

from const import DATA_DIR, MNIST_DIR, MAX_EPOCH, EXTRACT_EPOCHS, N_PER_CLASS, TSNE_PERPLEXITY, TSNE_ITER, PCA_DIM

torch.manual_seed(42)
np.random.seed(42)


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
        return self.fc_out(h4), [h1, h2, h3, h4]


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_mnist():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(MNIST_DIR, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(MNIST_DIR, train=False, download=True, transform=tf)
    return train, test

def balanced_indices(dataset, n_per_class=N_PER_CLASS):
    indices, counts = [], {i: 0 for i in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if counts[label] < n_per_class:
            indices.append(idx)
            counts[label] += 1
        if all(v >= n_per_class for v in counts.values()):
            break
    return indices

def extract(model, loader, device):
    model.eval()
    acts, labels = [[] for _ in range(4)], []
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
    dim = min(PCA_DIM, acts.shape[1], acts.shape[0] - 1)
    if dim < acts.shape[1]:
        acts = PCA(n_components=dim, random_state=42).fit_transform(acts)
    return TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITER,
        init='pca',
        learning_rate='auto',
        random_state=42
    ).fit_transform(acts)

def normalize(emb):
    emb = emb - emb.mean(0)
    return emb / (np.abs(emb).max() + 1e-8)

def procrustes_align(reference, target):
    ref_n = reference / (np.linalg.norm(reference) + 1e-8)
    tgt_n = target    / (np.linalg.norm(target)    + 1e-8)
    R, _  = orthogonal_procrustes(tgt_n, ref_n)
    return tgt_n @ R * np.linalg.norm(reference)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds, test_ds = load_mnist()
    test_loader  = DataLoader(Subset(test_ds, balanced_indices(test_ds)), batch_size=1000, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model     = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epoch_acts, epoch_stats, labels = {}, {}, None

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
            loss = criterion(model(data)[0], target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acts, _ = extract(model, test_loader, device)
        acc     = accuracy(model, test_loader, device)
        epoch_acts[epoch]  = acts
        epoch_stats[epoch] = {'loss': round(avg_loss, 4), 'acc': acc}
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | acc={acc:.1f}%")

    print("\nTraining complete. Running t-SNE …")

    # T1: inter-epoch evolution (layer 4 activations)
    t1_embs, ref_emb = {}, None
    for ep in EXTRACT_EPOCHS:
        print(f"  t-SNE epoch {ep} …", flush=True)
        emb = normalize(run_tsne(epoch_acts[ep][3]))
        if ref_emb is None:
            ref_emb = emb
        else:
            emb = normalize(procrustes_align(ref_emb, emb))
        t1_embs[ep] = emb

    t1_data = {
        "epochs":    EXTRACT_EPOCHS,
        "n_samples": len(labels),
        "layer":     "Layer 4 – 64 units (closest to output)",
        "stats":     {str(ep): epoch_stats[ep] for ep in EXTRACT_EPOCHS},
        "points": [
            {"id": int(i), "label": int(labels[i]),
             "positions": [{"epoch": ep, "x": float(t1_embs[ep][i, 0]), "y": float(t1_embs[ep][i, 1])}
                           for ep in EXTRACT_EPOCHS]}
            for i in range(len(labels))
        ]
    }

    with open(f'{DATA_DIR}/epochs.json', 'w') as f:
        json.dump(t1_data, f)
    print("Saved data/epochs.json")

    # T2: inter-layer evolution (final epoch activations)
    layer_names = ["Layer 1 – 512 units", "Layer 2 – 256 units", "Layer 3 – 128 units", "Layer 4 – 64 units"]
    final_acts  = epoch_acts[MAX_EPOCH]

    t2_embs, ref_emb = {}, None
    for li in range(4):
        print(f"  t-SNE layer {li+1} …", flush=True)
        emb = normalize(run_tsne(final_acts[li]))
        if ref_emb is None:
            ref_emb = emb
        else:
            emb = normalize(procrustes_align(ref_emb, emb))
        t2_embs[li] = emb

    t2_data = {
        "layers":    layer_names,
        "n_samples": len(labels),
        "epoch":     MAX_EPOCH,
        "points": [
            {"id": int(i), "label": int(labels[i]),
             "positions": [{"layer": li, "x": float(t2_embs[li][i, 0]), "y": float(t2_embs[li][i, 1])}
                           for li in range(4)]}
            for i in range(len(labels))
        ]
    }

    with open(f'{DATA_DIR}/layers.json', 'w') as f:
        json.dump(t2_data, f)
    print("Saved data/layers.json")


if __name__ == '__main__':
    main()
