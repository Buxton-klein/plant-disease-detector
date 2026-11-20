import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score
from src.dataset import ROOT, make_tf
from src.gradcam import load_model, short, SIZE

RW = ROOT / "data" / "plantdoc_tomato"


def run(model, classes, bs=32):
    tf = make_tf(SIZE, False)
    rows = []
    for cls_dir in sorted(RW.iterdir()):
        if cls_dir.is_dir() and cls_dir.name in classes:
            for p in sorted(cls_dir.glob("*.jpg")):
                rows.append((p, classes.index(cls_dir.name)))
    df = pd.DataFrame(rows, columns=["path", "y"])
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), bs):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in df["path"].iloc[i:i + bs]])
            probs.append(model(x).softmax(1))
    probs = torch.cat(probs).numpy()
    df["pred"] = probs.argmax(1)
    df["conf"] = probs.max(1)
    return df


if __name__ == "__main__":
    model, classes = load_model()
    df = run(model, classes)
    y, p = df["y"].to_numpy(), df["pred"].to_numpy()
    true_ids = sorted(set(y))
    print(f"images: {len(df)}  accuracy: {(y == p).mean():.3f}  "
          f"macro-F1 (classes present): {f1_score(y, p, labels=true_ids, average='macro'):.3f}")
    print("\nper-class recall:")
    for i in true_ids:
        m = y == i
        print(f"  {short(classes[i]):45s} n={m.sum():3d}  recall {(p[m] == i).mean():.2f}")
    print("\nhow often each class is predicted (all images):")
    for i, c in enumerate(classes):
        print(f"  {short(c):45s} {(p == i).sum():4d}")
    print(f"\nmean confidence: correct {df.conf[y == p].mean():.2f}, wrong {df.conf[y != p].mean():.2f}")

    cm = np.zeros((len(true_ids), len(classes)))
    for r, i in enumerate(true_ids):
        m = y == i
        cm[r] = np.bincount(p[m], minlength=len(classes)) / m.sum()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), [short(c) for c in classes], rotation=90)
    ax.set_yticks(range(len(true_ids)), [short(classes[i]) for i in true_ids])
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[r, c] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (PlantDoc)")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "plantdoc_confusion.png", dpi=150)
    plt.close()

    out = df.copy()
    out["path"] = out["path"].map(lambda q: Path(q).relative_to(ROOT).as_posix())
    out["true"] = [classes[i] for i in y]
    out["pred_name"] = [classes[i] for i in p]
    out.drop(columns=["y", "pred"]).to_csv(ROOT / "reports" / "plantdoc_predictions.csv", index=False)
