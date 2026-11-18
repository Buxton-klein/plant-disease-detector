import pandas as pd
from src.dataset import load_splits, make_loaders, SPLIT_DIR
from src.model import BaselineCNN
from src.train import set_seed, fit

if __name__ == "__main__":
    set_seed(42)
    train_df, val_df, test_df = load_splits()
    classes = (SPLIT_DIR / "classes.txt").read_text().split("\n")
    tr, va, te = make_loaders(train_df, val_df, test_df,
                              batch_size=32, num_workers=1, size=128)
    model = BaselineCNN(len(classes))
    hist = fit(model, tr, va, epochs=8, lr=1e-3, patience=3, ckpt="baseline_128.pt")
    pd.DataFrame(hist).to_csv("reports/baseline_128_history.csv", index=False)