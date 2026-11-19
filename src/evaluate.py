
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.dataset import ROOT, SPLIT_DIR, load_splits, make_loaders
from src.model import BaselineCNN


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval().to(device)
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def report(y, p, classes, name):
    short = [c.replace("Tomato___", "") for c in classes]
    print(classification_report(y, p, target_names=short, digits=3))
    cm = confusion_matrix(y, p, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short)), short, rotation=90)
    ax.set_yticks(range(len(short)), short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > 0.5 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name}: confusion matrix (row-normalized)")
    plt.tight_layout()
    (ROOT / "reports").mkdir(exist_ok=True)
    plt.savefig(ROOT / "reports" / f"{name}_confusion.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    classes = (SPLIT_DIR / "classes.txt").read_text().split("\n")
    tr, va, te = load_splits()
    _, _, test_loader = make_loaders(
        tr, va, te, batch_size=64, num_workers=0, size=128
    )
    model = BaselineCNN(len(classes))
    model.load_state_dict(
        torch.load(ROOT / "models" / "baseline_128.pt", map_location="cpu")
    )
    y, p = predict(model, test_loader)
    print(
        "test acc",
        round((y == p).mean(), 4),
        "macro-F1",
        round(f1_score(y, p, average="macro"), 4),
    )
    report(y, p, classes, "baseline_128")
    cm = confusion_matrix(y, p)
    short = [c.replace("Tomato___", "") for c in classes]
    pairs = [
        (cm[i, j], short[i], short[j])
        for i in range(len(short))
        for j in range(len(short))
        if i != j and cm[i, j] > 0
    ]

    print("\nTop confusions (count, true -> predicted):")
    for n, t, pr in sorted(pairs, reverse=True)[:8]:
        print(f"{n:3d}  {t} -> {pr}")
