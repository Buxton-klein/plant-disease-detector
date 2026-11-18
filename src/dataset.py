# src/dataset.py
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tomato"
SPLIT_DIR = ROOT / "data" / "splits"
IMG_EXT = {".jpg", ".jpeg", ".png"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def build_dataframe(data_dir=DATA_DIR):
    rows = []
    for cls_dir in sorted(Path(data_dir).iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.iterdir():
            if p.suffix.lower() in IMG_EXT:
                rows.append({"path": p.relative_to(ROOT).as_posix(),
                             "label": cls_dir.name})
    df = pd.DataFrame(rows)
    df["hash"] = df["path"].map(lambda p: _md5(ROOT / p))
    df = df.drop_duplicates("hash").reset_index(drop=True)
    classes = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    return df.drop(columns="hash"), classes


def make_splits(seed=42):
    df, classes = build_dataframe()
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["y"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["y"], random_state=seed)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
    (SPLIT_DIR / "classes.txt").write_text("\n".join(classes))
    return train_df, val_df, test_df, classes


def load_splits():
    return [pd.read_csv(SPLIT_DIR / f"{n}.csv") for n in ("train", "val", "test")]


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def make_tf(size, train):
    aug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
           transforms.ColorJitter(brightness=0.2, contrast=0.2)] if train else []
    return transforms.Compose([transforms.Resize((size, size)), *aug,
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class LeafDataset(Dataset):
    def __init__(self, df, transform=None, folder="tomato"):
        self.paths = [p.replace("data/tomato/", f"data/{folder}/", 1) for p in df["path"]]
        self.labels = df["y"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def make_loaders(train_df, val_df, test_df, batch_size=32, num_workers=4, size=128):
    folder = f"tomato_{size}"
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              persistent_workers=num_workers > 0)
    return (
        DataLoader(LeafDataset(train_df, make_tf(size, True), folder), shuffle=True, **kw),
        DataLoader(LeafDataset(val_df, make_tf(size, False), folder), shuffle=False, **kw),
        DataLoader(LeafDataset(test_df, make_tf(size, False), folder), shuffle=False, **kw),
    )
if __name__ == "__main__":
 train_df, val_df, test_df, classes = make_splits()
 print("train/val/test:", len(train_df), len(val_df), len(test_df))
 print("classes:", len(classes))
 print(test_df["label"].value_counts())