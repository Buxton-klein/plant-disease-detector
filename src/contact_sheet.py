import matplotlib
matplotlib.use("Agg")
import random
import matplotlib.pyplot as plt
from PIL import Image
from src.dataset import ROOT

RW = ROOT / "data" / "plantdoc_tomato"
dirs = sorted(d for d in RW.iterdir() if d.is_dir())
fig, axes = plt.subplots(len(dirs), 6, figsize=(12, 2.0 * len(dirs)))
for r, d in enumerate(dirs):
    files = sorted(d.glob("*.jpg"))
    for c, p in enumerate(random.Random(0).sample(files, 6)):
        axes[r, c].imshow(Image.open(p).convert("RGB"))
        axes[r, c].axis("off")
    axes[r, 0].set_title(d.name.replace("Tomato___", ""), fontsize=8, loc="left")
plt.tight_layout()
plt.savefig(ROOT / "reports" / "plantdoc_contact_sheet.png", dpi=90)
