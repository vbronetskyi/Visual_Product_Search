import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def get_embedding(image: Image.Image):
    """generate the embedding for a given image """
    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
        return embedding
    except Exception as e:
        raise RuntimeError(f"Error generating embedding: {e}")
