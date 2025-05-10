import os
import pickle
import random

# Parameters: limit number of pairs for faster training
MAX_POSITIVE_PAIRS = 20000  # max positive pairs sampled
MAX_NEGATIVE_PAIRS = 20000  # max negative pairs sampled

artifacts_dir = "data/artifacts"
# Load product info and kmeans model
with open(os.path.join(artifacts_dir, "product_info.pkl"), "rb") as f:
    product_info = pickle.load(f)
with open(os.path.join(artifacts_dir, "kmeans_model.pkl"), "rb") as f:
    kmeans = pickle.load(f)


clusters = {}
for item in product_info:
    eid = item["entity_id"]
    clusters.setdefault(eid, []).append(item["image_path"])

positive_pairs = []
for paths in clusters.values():
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            positive_pairs.append((paths[i], paths[j]))

if len(positive_pairs) > MAX_POSITIVE_PAIRS:
    positive_pairs = random.sample(positive_pairs, MAX_POSITIVE_PAIRS)

all_eids = list(clusters.keys())
negative_pairs = []
while len(negative_pairs) < MAX_NEGATIVE_PAIRS:
    e1, e2 = random.sample(all_eids, 2)
    p1 = random.choice(clusters[e1])
    p2 = random.choice(clusters[e2])
    negative_pairs.append((p1, p2))

pairs = positive_pairs + negative_pairs
labels = [1] * len(positive_pairs) + [-1] * len(negative_pairs)

combined = list(zip(pairs, labels))
random.shuffle(combined)
pairs, labels = zip(*combined)

with open(os.path.join(artifacts_dir, "pairs.pkl"), "wb") as f:
    pickle.dump({"pairs": list(pairs), "labels": list(labels)}, f)

print(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs ({len(pairs)} total).")
