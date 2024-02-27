import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torch.nn.functional import normalize
from torch import nn

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
    y = normalize(y) if norm else y
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

    def compute(self, norm=False):
        # parse inputs
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return lalign(preds, target, self.alpha, norm)


# And torchmetrics variants:
class Uniformity(Metric):
    def __init__(self, t=2, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.t = t

    def update(self, preds: Tensor) -> None:
        self.preds.append(preds)

    def compute(self, norm=False):
        # parse inputs
        preds = dim_zero_cat(self.preds)
        return lunif(preds, self.t, norm)

#TBD
def compute_cosines(embedding, embeddings):
    #for k,v in embeddings.items():
    cos0 = torch.nn.CosineSimilarity(dim=1)
    return cos0(embedding.unsqueeze(0).repeat(embeddings.shape[0],1), embeddings)

def get_rank(x, indices):
    vals = x[range(len(x)), indices]
    return (x > vals[:, None]).long().sum(1)

def get_rank_metrics(embeddings, modality, fusion="fusion"):
    c=list()
    #xc=list()
    idx = list()
    batch_size = embeddings[modality].shape[0]
    #print(e[modality].shape)
    for i in range(batch_size):
        mx=m[modality][i]
        if not mx:
            pass
        idx.append(i)
        x=embeddings[modality][i,:] #.shape
        #xx=e['pt']
        #xc.append(compute_recall(x,xx).topk(10))
        y=embeddings[fusion]#.shape
        c.append(compute_cosines(x,y))
        #print(c[i].topk(25))
        #_=plt.hist(c.cpu(), bins=1000, log=True)
    ranks = get_rank(torch.stack(c), torch.tensor(idx))
    median_rank = ranks.median()
    r1 = sum(ranks == 0)/len(ranks)
    r5 = sum(ranks < 5)/len(ranks)
    r10 = sum(ranks < 10)/len(ranks)
    return median_rank, r1, r5, r10