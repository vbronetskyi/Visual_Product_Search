import os, random, zipfile

ZIP_PATH = "retail-786k_256.zip"
OUT_DIR = "data/dataset/retail-786k_256_sample"
SAMPLE_SIZE = 10000

os.makedirs(OUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    all_imgs = [
        name for name in z.namelist()
        if name.startswith("retail-786k_256/train/") and name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    sample = random.sample(all_imgs, min(SAMPLE_SIZE, len(all_imgs)))

    for member in sample:
        z.extract(member, OUT_DIR)

print(f"Extracted {len(sample)} files into {OUT_DIR}")
