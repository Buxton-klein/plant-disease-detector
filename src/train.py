import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from src.dataset import ROOT


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            preds.append(out.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds), torch.cat(targets)
    return {
        "loss": total / len(targets),
        "acc": (preds == targets).float().mean().item(),
        "macro_f1": f1_score(targets, preds, average="macro"),
    }


def fit(model, train_loader, val_loader, epochs=10, lr=1e-3,
        patience=4, ckpt="baseline.pt", device="cpu"):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad, history = 0.0, 0, []
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        print(f"ep {ep:02d} | train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} f1 {va['macro_f1']:.3f}")
        if va["macro_f1"] > best:
            best, bad = va["macro_f1"], 0
            (ROOT / "models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "models" / ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break
    return history