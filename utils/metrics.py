import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torch.nn.functional import normalize

#from torchmetrics.utilities import dim_zero_cat

# Below is directly from Wang and Isola 2021 Understanding Contrastive Rep Learning...

# bsz : batch size (number of positive pairs)
# d : latent dim
# x : Tensor, shape=[bsz, d]
# latents for one side of positive pairs
# y : Tensor, shape=[bsz, d]
# latents for the other side of positive pairs
# lam : hyperparameter balancing the two losses

def lalign(x, y, alpha=2, norm=True):
    x = normalize(x) if norm else x
    y = normalize(y) if norm else x
    return (x - y).norm(dim=1).pow(alpha).mean()


def lunif(x, t=2, norm=True):
    x = normalize(x) if norm else x
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def wang_loss(x, y, lam=1.0, alpha=2, t=2):
    return lalign(x, y, alpha) + lam * (lunif(x, t) + lunif(y, t)) / 2


# And torchmetrics variants:
class Alignment(Metric):
    def __init__(self, alpha=2, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.alpha=alpha

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

    def compute(self):
        # parse inputs
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return lalign(preds, target, self.alpha)


# And torchmetrics variants:
class Uniformity(Metric):
    def __init__(self, t=2, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.t = t

    def update(self, preds: Tensor) -> None:
        self.preds.append(preds)

    def compute(self):
        # parse inputs
        preds = dim_zero_cat(self.preds)
        return lunif(preds, self.t)
