import copy

import torch


class ModelEMA:
    """Maintain an EMA copy of model weights for stable evaluation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd:
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)
            else:
                esd[k].copy_(msd[k])
