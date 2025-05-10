import torch, clip

class SiameseNetwork(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.backbone = clip_model

    def forward(self, x1, x2):
        e1 = self.backbone.encode_image(x1)
        e2 = self.backbone.encode_image(x2)
        # L2-нормалізація
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = e2 / e2.norm(dim=-1, keepdim=True)
        return e1, e2
