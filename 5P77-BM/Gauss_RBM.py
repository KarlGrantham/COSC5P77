import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Gauss_RBM(nn.Module):
    r"""Restricted Boltzmann Machine.

    Args:
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden lsum_hj_Wijayer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_vis=784, n_hid=128, k=1):
        """Create a Gauss_RBM."""
        super(Gauss_RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(1, n_vis))#visable bias term, b^v
        self.h = nn.Parameter(torch.randn(1, n_hid))#hidden bias term, b^h
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))#weight matrix, W_i_j
        self.cov = nn.Parameter(torch.abs(torch.randn(1, n_vis)))#diagonal covariance matrix, implemented by a 1D tensor
        self.k = k
    def absolute_cov(self):
        self.cov = torch.nn.Parameter(torch.abs(self.cov))
        
    def visible_to_hidden(self, v):
        r"""Conditional sampling a hidden variable given a visible variable.

        Args:
            v (Tensor): The visible variable.

        Returns:
            Tensor: The hidden variable.

        """
        v_over_sigma = torch.div(v,self.cov)#element-wise division of v over sigma
        #same as in the binary-visibles case except here the real-valued visible activity
        #by the reciprocal of its standard deviation v_i is scaled σ_i
        #note: not using W^T here because of order of vector-matrix multiplication
        p = torch.sigmoid(F.linear(v_over_sigma,self.W,self.h))#1/ 1+e^-((v/sigma^2) * W + b^h)
        #since the normal part only corresponds to sampling the visible part,
        #we still sample from the hidden part using a bern dist
        #thus, the code for hidden_to_visible remains the same.
        return p.bernoulli()

    
    def hidden_to_visible(self, h):
        r"""Conditional sampling a visible variable given a hidden variable.

        Args:
            h (Tensor): The hidden variable.

        Returns:
            Tensor: The visible variable.

        """
        
        sum_hj_Wij = torch.matmul(h, self.W)#h_j * W_i_j
        mean_vec = torch.add(self.v, sum_hj_Wij)  #b^v + h_j * W_i_j
        #print(mean_vec.shape)
        self.absolute_cov()#required for torch.normal to work
        result = torch.normal(mean_vec, self.cov)#N(mu, sigma^2)
        #print(result.shape)
        #quit()
        
        #Resuting vector is expected to be normalized to [0,1]
        #Apparently, normalization to [0,1] has to be done manually. See:
        #https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
        result -= result.min(1, keepdim=True)[0]
        result /= result.max(1, keepdim=True)[0]
        return result

    def free_energy(self, v):
        r"""Free energy function.

        Args:
            v (Tensor): The visible variable.

        Returns
            FloatTensor: The free energy value.

        """
        #operations in strange order so that the shapes match up
        vi_sigma = torch.div(v,self.cov)#v_i/sigma^2_i
        Wij_vi_sigma = torch.matmul(self.W, vi_sigma.t())
        #Wij_vi = torch.matmul(self.W, v.t())
        #Wij_vi_sigma = torch.div(Wij_vi, self.cov)#(W_i_j * v_i)/sigma_i^2
        h = self.visible_to_hidden(v)#needed for gradient
        h_sum = torch.matmul(h, Wij_vi_sigma).sum()#sum_j hj * (W_i_j * v_i)/sigma_i^2
        sigma3 = torch.matmul(torch.diag(self.cov), torch.sqrt(self.cov))#sigma^3, implemented by conversion to diag matrix, then matrix-vector multiplication of sigma^2 * sigma.
        vi_bvi_sigma3 = torch.div(torch.square((torch.sub(v,self.v))), sigma3)#((v_i - b_i^v)^2)/sigma^3
        result = torch.sub(vi_bvi_sigma3, torch.mean(h_sum))#the result, using torch.mean to get a true scalar out of a 1x1 tensor
        
        return torch.mean(result)#expected value, therefore mean of tensor

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
