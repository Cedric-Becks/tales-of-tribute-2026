import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import TalesDataset, collate_fn
from network import TalesValueNet

DATA_DIR   = 'data'
VAL_SPLIT  = 0.1
BATCH_SIZE = 256
EPOCHS     = 30
LR         = 1e-3
SAVE_PATH  = 'model_best.pt'
DEVICE     = 'mps' if torch.backends.mps.is_available() else 'cpu'


def train():
    print("Loading dataset...")
    dataset = TalesDataset(DATA_DIR)
    n       = len(dataset)
    n_val   = max(1, int(n * VAL_SPLIT))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"  {n_train} train / {n_val} val samples")
    print(f"  Vocab size: {dataset.vocab_size}   Device: {DEVICE}")

    model     = TalesValueNet(dataset.vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for zones, scalars, patrons, outcomes in train_loader:
            zones    = {k: (v[0].to(DEVICE), v[1].to(DEVICE)) for k, v in zones.items()}
            scalars  = scalars.to(DEVICE)
            patrons  = patrons.to(DEVICE)
            outcomes = outcomes.to(DEVICE)

            pred = model(zones, scalars, patrons)
            loss = criterion(pred, outcomes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(outcomes)

        # --- validate ---
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        with torch.no_grad():
            for zones, scalars, patrons, outcomes in val_loader:
                zones    = {k: (v[0].to(DEVICE), v[1].to(DEVICE)) for k, v in zones.items()}
                scalars  = scalars.to(DEVICE)
                patrons  = patrons.to(DEVICE)
                outcomes = outcomes.to(DEVICE)

                pred         = model(zones, scalars, patrons)
                val_loss    += criterion(pred, outcomes).item() * len(outcomes)
                val_correct += ((pred > 0.5) == (outcomes > 0.5)).sum().item()

        train_loss /= n_train
        val_loss   /= n_val
        val_acc     = val_correct / n_val * 100

        marker = " ← saved" if val_loss < best_val_loss else ""
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.1f}%{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state': model.state_dict(), 'vocab': dataset.vocab}, SAVE_PATH)

    print(f"\nBest val loss: {best_val_loss:.4f} — saved to {SAVE_PATH}")


if __name__ == '__main__':
    train()
