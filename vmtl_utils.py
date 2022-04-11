import torch
import torch.nn.functional as f


# local_reparameterization trick for sampling
def local_reparameterize_softplus(mu, var, repeat):
    '''
       the size of input(mu) is d_out * d_in, we sample one eps for each column.
       the size of output is bs * d_out * d_in.
    '''                 
    eps = torch.cuda.FloatTensor(repeat, mu.shape[0], mu.shape[1]).normal_(0,1)
    sigma = var.sqrt()
    sigma = sigma.expand(repeat, sigma.shape[0], sigma.shape[1])
    mu = mu.expand(repeat, mu.shape[0], mu.shape[1])  
    return mu + sigma*eps

# kl term for VMTL
def kl_criterion_softplus(mu_e, var_e, mu_p, var_p):
    var_e = var_e + 1e-6
    var_p = var_p + 1e-6
    component1 = torch.log(var_p) - torch.log(var_e)
    component2 = var_e / var_p
    component3 = (mu_p - mu_e).pow(2)/ var_p
    KLD = 0.5 * torch.sum((component1 -1 +component2 +component3),1)
    return KLD

# gumbel
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = torch.log(f.softmax(logits, 1) + 1e-20) + sample_gumbel(logits.size())
    return f.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard