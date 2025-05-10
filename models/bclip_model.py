import torch, clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("data/artifacts/siamese_clip.pt", map_location=device))
model.eval()

def get_embedding(image: Image.Image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    return emb / emb.norm(dim=-1, keepdim=True)
