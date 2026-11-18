from PIL import Image
from src.dataset import ROOT, DATA_DIR, IMG_EXT

def resize_all(size=128):
    out_root = ROOT / "data" / f"tomato_{size}"
    n = 0
    for p in DATA_DIR.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            out = out_root / p.relative_to(DATA_DIR)
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR).save(out, quality=95)
                n += 1
    print("wrote", n)

if __name__ == "__main__":
    resize_all(128)