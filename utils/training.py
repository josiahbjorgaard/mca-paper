import torch

def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")

def copy_batch(obj):
  if torch.is_tensor(obj):
    return obj.detach().clone()
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = copy_batch(v)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(copy_batch(v))
    return res
  else:
    raise TypeError("Invalid type for copy_to")

def count_parameters(model,print_summary=False):
    n_param_embedding = 0
    n_param_nonembedding = 0
    for n,p in model.named_parameters():
        if p.requires_grad:
            if print_summary:
                print(f"{n}:{p.numel()/10**6}M")
            if 'embedding' in n:
                n_param_embedding+=p.numel()
            else:
                n_param_nonembedding+=p.numel()
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_param_embedding, n_param_nonembedding

def get_param_norm(model,norm_type=2.0):
    norm_type = float(norm_type)
    parameters = model.parameters()
    local_norm = torch.DoubleTensor([0.0]).to(next(iter(parameters)).device)
    grads_for_norm = []
    for param in parameters:
        param_norm = torch.norm(param.detach(), norm_type)
        local_norm += param_norm ** norm_type
    total_norm = local_norm**(1.0 / norm_type)
    return total_norm

def get_grad_norm(model,norm_type=2.0):
    norm_type = float(norm_type)
    parameters = model.parameters()
    local_norm = torch.FloatTensor([float(0.0)]).to(next(iter(parameters)).device)
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        if grad_not_none:
            grad = param.grad.detach()
            grad_norm = torch.norm(grad, norm_type)
            local_norm += grad_norm ** norm_type
    total_norm = local_norm**(1.0 / norm_type)
    return total_norm
