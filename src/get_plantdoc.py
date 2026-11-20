from pathlib import Path
from datasets import load_dataset

MAP = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
}
out = Path("data/plantdoc_tomato")
ds = load_dataset("Project-AgML/plant_doc_classification")
print(ds)
counts = {}
for split in ds:
    names = ds[split].features["label"].names
    print(split, [x for x in names if "omato" in x])
    for i, ex in enumerate(ds[split]):
        name = names[ex["label"]]
        if name in MAP:
            d = out / MAP[name]
            d.mkdir(parents=True, exist_ok=True)
            ex["image"].convert("RGB").save(d / f"{split}_{i}.jpg", quality=95)
            counts[MAP[name]] = counts.get(MAP[name], 0) + 1
print(counts)
