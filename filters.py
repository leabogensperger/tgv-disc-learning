import numpy as np
import matplotlib.pyplot as plt
import imageio

import torch
import torch.fft
import torch.nn.functional as F
import sys

try:
    from pad2d_op import pad2d, pad2dT
except ImportError:
    pass

class IdFilter:
    # identity filter to have same structure for reference TGV
    def apply(self, arg):
        return arg
    
    def apply_adjoint(self, arg):
        return arg

def fun_padT(y_padded, pad, mode, value=0.0):
    """ Get the gradient of the padding operator"""
    N,C,H,W = y_padded.shape
    x = y_padded.new_zeros([N, C, H-pad-pad, W-pad-pad])
    x = torch.autograd.Variable(x, requires_grad=True)
    x_pseudo_padded = torch.nn.functional.pad(x,  (pad,pad,pad,pad), mode=mode, value=value) 
    # Inefficient: Forward pass required, also not TorchScript jit - scriptable
    return torch.autograd.grad(outputs=x_pseudo_padded,  inputs=x,
                        grad_outputs=y_padded,
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True)[0]

class BilinearFilter:
    def __init__(self, n, d, ker, pad_mode, adjoint_mode, dtype):
        self.n = n
        self.d = d
        self.pad_mode = pad_mode

        # sample from uniform distribution (-1/sqrt(k),1/sqrt(k)) with k=groups/(cin*ker*ker) ---- pytorch init
        self.weights = torch.from_numpy(np.random.uniform(low=-np.sqrt(d/(d*ker*ker)),high=np.sqrt(d/(d*ker*ker)),size=(d*n,1,ker,ker))).type(dtype)
        self.weights = self.weights + 1/(ker*ker)*(1-self.weights.sum(axis=(-2,-1), keepdims=True))

        self.adjoint_mode = adjoint_mode # V1: using autograd, V2: zero-padding, V3: CUDA op with more padding options
        padW = int((self.weights.shape[-1]-1)/2)
        padH = int((self.weights.shape[-2]-1)/2)
        assert padW == padH # for now only use NxN images
        self.pad_sz = padW

    def apply(self, arg):
        # L: (B,2,M1,M2) -> (B,6,M1,M2), K: (B,3,M1,M2) -> (B,3*nK,M1,M2)
        if self.adjoint_mode == 1: # V1 -- using autograd
            arg_pad = F.pad(arg, (self.pad_sz,self.pad_sz,self.pad_sz,self.pad_sz), self.pad_mode)
            arg_conv = F.conv2d(arg_pad, self.weights.to(arg.device), groups=self.d)   
        elif self.adjoint_mode == 2: # V2 -- zero-padding
            arg_conv = F.conv2d(arg, self.weights.to(arg.device),padding=(self.pad_sz,self.pad_sz), groups=self.d)
        elif self.adjoint_mode == 3: # V3 -- CUDA op with more padding options
            arg_pad = pad2d(arg,padding=[self.pad_sz,self.pad_sz,self.pad_sz,self.pad_sz],mode=self.pad_mode)
            arg_conv = F.conv2d(arg_pad, self.weights.to(arg.device),groups=self.d)
        else:
            raise ValueError('Unknown pad/conv mode chosen!')

        return arg_conv

    def apply_adjoint(self, arg):
        # L: (B,6,M1,M2) -> (B,2,M1,M2), K: (B,3*nK,M1,M2) -> (B,3,M1,M2)
        if self.adjoint_mode == 1: # V1
            arg_cropT = torch.nn.functional.pad(arg, pad=(self.pad_sz,self.pad_sz,self.pad_sz,self.pad_sz), mode="constant", value=0)
            arg_cropT_convT = torch.nn.functional.conv_transpose2d(arg_cropT, weight=self.weights.to(arg.device), padding=0, groups=self.d)[:,:,self.pad_sz:-self.pad_sz,self.pad_sz:-self.pad_sz]
            return fun_padT(arg_cropT_convT, pad=self.pad_sz, mode=self.pad_mode)
        elif self.adjoint_mode == 2: # V2
            return F.conv_transpose2d(arg, self.weights.to(arg.device),padding=(self.pad_sz,self.pad_sz), groups=self.d)  
        elif self.adjoint_mode == 3: # V3
            arg_conv = F.conv_transpose2d(arg, self.weights.to(arg.device), groups=self.d)
            return pad2dT(arg_conv,padding=[self.pad_sz,self.pad_sz,self.pad_sz,self.pad_sz],mode=self.pad_mode)  
        else:
            raise ValueError('Unknown pad/conv mode chosen!')

    def check_adjointness(self, check_it, dim, dtype, device):
        print('\nAdjointness Check Interpolation Operator:')
        for _ in range(check_it):
            divs = torch.randn(1,self.d,dim,dim).type(dtype).to(device)
            vL = torch.randn(1,self.d*self.n,dim,dim).type(dtype).to(device)
            lhs = (self.apply(divs)*vL).sum()
            rhs = (self.apply_adjoint(vL)*divs).sum()
            print('\tAjointness holds: %s (abs. diff. %.12f)' %(torch.allclose(lhs,rhs),torch.abs(lhs-rhs)))

if __name__ == '__main__':
    device = 'cuda'
    dtype = torch.float32
    check_it = 10
    dim = 100 
    adjoint_mode = 3 # padding options for adjoint linear operator
 
    # bilinear op K, L 
    BilinearFilter(n=1, d=3, pad_mode='reflect', ker=3, adjoint_mode=adjoint_mode, dtype=dtype).check_adjointness(check_it=check_it, dim=dim, dtype=dtype, device=device)
    BilinearFilter(n=3, d=2, pad_mode='reflect', ker=3, adjoint_mode=adjoint_mode, dtype=dtype).check_adjointness(check_it=check_it, dim=dim, dtype=dtype, device=device)
