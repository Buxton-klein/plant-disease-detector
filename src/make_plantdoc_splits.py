import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset import ROOT, SPLIT_DIR

RW = ROOT / "data" / "plantdoc_tomato"
rows = [(p.relative_to(ROOT).as_posix(), d.name)
        for d in sorted(RW.iterdir()) if d.is_dir()
        for p in sorted(d.glob("*.jpg"))]
df = pd.DataFrame(rows, columns=["path", "label"])
classes = (SPLIT_DIR / "classes.txt").read_text().split("\n")
df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
train, rest = train_test_split(df, test_size=0.40, stratify=df["y"], random_state=42)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["y"], random_state=42)
for name, d in [("train", train), ("val", val), ("test", test)]:
    d.to_csv(SPLIT_DIR / f"plantdoc_{name}.csv", index=False)
    print(name, len(d))
print(test["label"].value_counts())
