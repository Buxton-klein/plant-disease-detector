import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from src.dataset import ROOT, SPLIT_DIR, LeafDataset, load_splits, make_tf
from src.model import build_resnet18

SIZE = 224


def short(name):
    return name.replace("Tomato___", "")


def load_model(ckpt="resnet18_ft.pt"):
    classes = (SPLIT_DIR / "classes.txt").read_text().split("\n")
    model = build_resnet18(len(classes), pretrained=False)
    model.load_state_dict(torch.load(ROOT / "models" / ckpt, map_location="cpu"))
    return model.eval(), classes


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        if output.requires_grad:
            self.acts = output.detach()
            output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        cls = out.argmax(1).item()
        out[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))[0]
        cam = (cam / (cam.max() + 1e-8)).numpy()
        prob = out.softmax(1)[0, cls].item()
        return cam, cls, prob


def overlay(img, cam):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
    heat = plt.get_cmap("jet")(np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(img.resize((SIZE, SIZE))) / 255.0
    return 0.55 * base + 0.45 * heat


def explain(cam_tool, row):
    img = Image.open(ROOT / row["path"]).convert("RGB")
    x = make_tf(SIZE, False)(img).unsqueeze(0)
    cam, cls, prob = cam_tool(x)
    return img.resize((SIZE, SIZE)), overlay(img, cam), cls, prob


def samples_grid(model, classes, test_df, per_class=2):
    cam_tool = GradCAM(model, model.layer4)
    fig, axes = plt.subplots(len(classes), per_class * 2,
                             figsize=(2.6 * per_class * 2, 2.6 * len(classes)))
    for ax in axes.flat:
        ax.axis("off")
    for i, c in enumerate(classes):
        rows = test_df[test_df["y"] == i].sample(per_class, random_state=0)
        for k, (_, row) in enumerate(rows.iterrows()):
            img, vis, cls, prob = explain(cam_tool, row)
            axes[i, 2 * k].imshow(img)
            axes[i, 2 * k].set_title(short(c), fontsize=7)
            axes[i, 2 * k + 1].imshow(vis)
            axes[i, 2 * k + 1].set_title(f"pred: {short(classes[cls])} {prob:.2f}", fontsize=7)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "gradcam_samples.png", dpi=110)
    plt.close()


def predict_all(model, df, bs=64):
    loader = DataLoader(LeafDataset(df, make_tf(SIZE, False)), batch_size=bs)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x).argmax(1))
    return torch.cat(preds).numpy()


def errors_grid(model, classes, test_df, preds, n=8):
    wrong = test_df[preds != test_df["y"].to_numpy()]
    wrong = wrong.sample(min(n, len(wrong)), random_state=0)
    cam_tool = GradCAM(model, model.layer4)
    rows_n = int(np.ceil(len(wrong) / 2))
    fig, axes = plt.subplots(rows_n, 4, figsize=(10.4, 2.8 * rows_n))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, row) in enumerate(wrong.iterrows()):
        r, c = divmod(k, 2)
        img, vis, cls, prob = explain(cam_tool, row)
        axes[r, 2 * c].imshow(img)
        axes[r, 2 * c].set_title("true: " + short(classes[row["y"]]), fontsize=7)
        axes[r, 2 * c + 1].imshow(vis)
        axes[r, 2 * c + 1].set_title(f"pred: {short(classes[cls])} {prob:.2f}", fontsize=7)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "gradcam_errors.png", dpi=110)
    plt.close()


def corner_probe(model, classes, test_df, patch=64, bs=64):
    tf = make_tf(SIZE, False)
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_df), bs):
            paths = test_df["path"].iloc[i:i + bs]
            x = torch.stack([tf(Image.open(ROOT / p).convert("RGB").crop((0, 0, patch, patch)))
                             for p in paths])
            preds.append(model(x).argmax(1))
    preds = torch.cat(preds).numpy()
    y = test_df["y"].to_numpy()
    print(f"corner-only accuracy: {(preds == y).mean():.3f} "
          f"(always-biggest-class: {np.bincount(y).max() / len(y):.3f}, uniform guess: {1 / len(classes):.2f})")
    for i, c in enumerate(classes):
        print(f"  {short(c):45s} {(preds[y == i] == i).mean():.2f}")
    fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))
    for ax, p in zip(axes, test_df["path"].iloc[:8]):
        ax.imshow(Image.open(ROOT / p).convert("RGB").crop((0, 0, patch, patch)))
        ax.axis("off")
    plt.savefig(ROOT / "reports" / "corner_crops.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    model, classes = load_model()
    _, _, test_df = load_splits()
    (ROOT / "reports").mkdir(exist_ok=True)
    samples_grid(model, classes, test_df)
    preds = predict_all(model, test_df)
    y = test_df["y"].to_numpy()
    print("test acc:", round((preds == y).mean(), 4), "errors:", int((preds != y).sum()))
    errors_grid(model, classes, test_df, preds)
    corner_probe(model, classes, test_df)
