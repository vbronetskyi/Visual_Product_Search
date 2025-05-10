import os, pickle
from PIL import Image, UnidentifiedImageError
import torch, clip
import numpy as np
from sklearn.cluster import KMeans

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

dataset_dir = r"data/dataset/retail-786k_256_sample/retail-786k_256/train"
artifacts_dir = "data/artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

embeddings = []
product_info = []

for root, _, files in os.walk(dataset_dir):
    entity_id = os.path.basename(root)
    for fn in files:
        if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif")):
            path = os.path.join(root, fn)
            try:
                img = Image.open(path).convert("RGB")
            except UnidentifiedImageError:
                continue

            inp = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(inp).cpu().numpy().flatten()
            embeddings.append(emb)
            product_info.append({
                "image_path": path,
                "entity_id": int(entity_id),
                "filename": fn
            })

embeddings = np.vstack(embeddings)

# KMeans for embeddings
n_clusters = len({pi["entity_id"] for pi in product_info})
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(embeddings)

with open(os.path.join(artifacts_dir, "embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)
with open(os.path.join(artifacts_dir, "product_info.pkl"), "wb") as f:
    pickle.dump(product_info, f)
with open(os.path.join(artifacts_dir, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)

print("Precompute done: embeddings + kmeans saved.")
