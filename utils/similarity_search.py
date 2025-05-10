import os, pickle, numpy as np
from functools import lru_cache

def cosine(a,b,eps=1e-10):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+eps)

@lru_cache(1)
def load_data():
    art = "data/artifacts"
    with open(os.path.join(art,"embeddings.pkl"),"rb") as f: embs = pickle.load(f)
    with open(os.path.join(art,"product_info.pkl"),"rb") as f: info = pickle.load(f)
    return embs, info

def find_similar(query_emb, top_n=5):
    embs, info = load_data()
    q = query_emb.cpu().numpy().flatten()
    sims = [cosine(q, e) for e in embs]
    idxs = np.argsort(sims)[::-1][:top_n]
    return [{"image_path": info[i]["image_path"], "score": sims[i]} for i in idxs]
