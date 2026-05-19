import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    """L1 perceptual loss on shallow VGG16 features for [0, 1] RGB inputs.
    Large inputs are bilinearly resized to keep max side <= max_input_side
    to reduce memory usage.
    """

    def __init__(self, layers: int = 23, max_input_side: int = 256):
        super().__init__()
        self.max_input_side = max_input_side
        try:
            from torchvision.models import VGG16_Weights, vgg16

            w = VGG16_Weights.IMAGENET1K_V1
            body = vgg16(weights=w).features[:layers]
        except Exception:
            from torchvision.models import vgg16

            body = vgg16(pretrained=True).features[:layers]
        self.net = body.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        # Running VGG at full resolution scales roughly with O(HW) memory;
        # limiting the longest side helps prevent OOM (default: 256).
        if self.max_input_side and self.max_input_side > 0:
            _, _, h, w = pred.shape
            m = max(h, w)
            if m > self.max_input_side:
                scale = self.max_input_side / float(m)
                nh = max(1, int(round(h * scale)))
                nw = max(1, int(round(w * scale)))
                pred = F.interpolate(pred, size=(nh, nw), mode="bilinear", align_corners=False)
                target = F.interpolate(target, size=(nh, nw), mode="bilinear", align_corners=False)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.net(pred), self.net(target))
