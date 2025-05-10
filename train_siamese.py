import os, pickle, random, time
from PIL import Image
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from torch.nn import CosineEmbeddingLoss
from torch.optim import Adam

ARTIFACTS_DIR = "data/artifacts"
PAIRS_PATH = os.path.join(ARTIFACTS_DIR, "pairs.pkl")
SAMPLE_SIZE = 10000 # limit total pairs for faster training, if needed
BATCH_SIZE = 32
EPOCHS = 7   # fewer epochs for quick iteration
LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_INTERVAL = 500 # log every N batches

with open(PAIRS_PATH, "rb") as f:
    data = pickle.load(f)
pairs = data["pairs"]
labels = data["labels"]

if len(pairs) > SAMPLE_SIZE:
    sampled_indices = random.sample(range(len(pairs)), SAMPLE_SIZE)
    pairs = [pairs[i] for i in sampled_indices]
    labels = [labels[i] for i in sampled_indices]

class SiameseDataset(Dataset):
    def __init__(self, pairs, labels, preprocess):
        self.pairs = pairs
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        img1 = self.preprocess(Image.open(p1).convert("RGB"))
        img2 = self.preprocess(Image.open(p2).convert("RGB"))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img1, img2, label

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

class SiameseNetwork(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x1, x2):
        e1 = self.backbone.encode_image(x1)
        e2 = self.backbone.encode_image(x2)
        #L2 normalization
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = e2 / e2.norm(dim=-1, keepdim=True)
        return e1, e2

model = SiameseNetwork(clip_model).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CosineEmbeddingLoss(margin=0.5)

dataset = SiameseDataset(pairs, labels, preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (img1, img2, lbl) in enumerate(loader, start=1):
        img1, img2, lbl = img1.to(DEVICE), img2.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        e1, e2 = model(img1, img2)
        loss = criterion(e1, e2, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0 or batch_idx == len(loader):
            elapsed = time.time() - start_time
            avg_loss = total_loss / batch_idx
            print(f"Epoch {epoch}/{EPOCHS}  Batch {batch_idx}/{len(loader)}  "
                  f"Elapsed: {elapsed:.1f}s  Avg Loss: {avg_loss:.4f}")

    epoch_time = time.time() - start_time
    print(f"--- Epoch {epoch} completed in {epoch_time:.1f}s  "
          f"Average Loss: {total_loss/len(loader):.4f} ---")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
torch.save(model.backbone.state_dict(), os.path.join(ARTIFACTS_DIR, "siamese_clip_finetuned.pt"))
print("Fine-tuned model saved to", ARTIFACTS_DIR)
