import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math


class BM(nn.Module):
    r"""Boltzmann Machine.

    Args:
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_vis=784, n_hid=128, k=1):
        """Create a BM."""
        super(BM, self).__init__()
        self.a = nn.Parameter(torch.randn(1, n_vis))#visable bias term, a
        self.b = nn.Parameter(torch.randn(1, n_hid))#hidden bias term, b
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))#weight matrix
        self.U = nn.Parameter(torch.randn(n_vis, n_vis))#homogenous ineraction matrix for v
        self.V = nn.Parameter(torch.randn(n_hid, n_hid))#homogenous ineraction matrix for h
        self.mu = n.Parameter(torch.randn(1, n_hid))#the approximation of the hidden, since it's intractable
        self.k = k

    def expectation_energy(self, v, mu):
        a_t_v = torch.matmul(a.t(),v)#a^T * x
        b_t_mu = torch.matmul(a.t(),v)#b^T * mu
        homo_v = torch.matmul(v.t(),torch.matmul(self.U,v))#x^T * U * x
        homo_mu = torch.matmul(mu.t(),torch.matmul(self.V,mu))#mu^T * V * mu
        hetr = torch.matmul(v.t(),torch.matmul(self.W,mu))#x^T * W * mu
        #thus, we end up with a whole bunch of scalars which we will sum.
        #therefore use torch mean to get true scalar out of 1x1 tensor.
        #Also apply multiples to homo interactions.
        a_t_v = torch.mean(a_t_v)
        b_t_mu = torch.mean(b_t_mu)
        homo_v = 0.5 * torch.mean(homo_v)
        homo_mu = 0.5 * torch.mean(homo_mu)
        hetr = torch.mean(hetr)
        
        return (a_t_v + b_t_mu + homo_v + homo_mu + hetr)
        
    def entorpy(self, mu):
        entropy_sum = 0
        for mu_i in mu:
            entropy_sum = entropy_sum + mu_i * math.log1p(mu_i) + (1 - mu_i) * (math.log1p(1 - mu_i))
        
        return entropy_sum
          
    def estimate_mu(self, v):
        new_mu = []
        for k, mu_k in enumerate(self.mu)
            new_mu.append(torch.sigmoid(self.b[k] + (torch.inner(self.V[k],self.mu) + torch.inner(self.W.t()[k], v))
        self.mu = torch.tensor(new_mu)
        
    def forward(self, v):
        r"""Compute the real and generated examples.

        Args:
            v (Tensor): The visible variable.

        Returns:
            (Tensor, Tensor): The real and generagted variables.

        """
        h = self.visible_to_hidden(v)
        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(h)
            h = self.visible_to_hidden(v_gibb)
        return v, v_gibb
