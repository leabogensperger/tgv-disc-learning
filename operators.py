import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.fft
import torch.nn.functional as F
import sys

try:
    from pad2d_op import pad2d, pad2dT
except ImportError:
    pass

class D:
    def __init__(self,h=1.):
        # finite forward difference operator
        self.h = h

    def apply(self, arg):
        # (B,1,M1,M2) -> (B,2,M1,M2)
        B,_,M1,M2 = arg.shape
        ret = torch.zeros((B,2,M1,M2), dtype=arg.dtype, device=arg.device)
        
        ret[:,0,:,:-1] = arg[:,0,:,1:] - arg[:,0,:,:-1]
        ret[:,1,:-1,:] = arg[:,0,1:,:] - arg[:,0,:-1,:]
        return ret/self.h

    def apply_adjoint(self, arg):
        # (B,2,M1,M2) -> (B,1,M1,M2)
        B,_,M1,M2 = arg.shape
        ret = torch.zeros((B,1,M1,M2), dtype=arg.dtype, device=arg.device)

        ret[:,0] = F.pad(arg[:,0,:,:-1],(1,0,0,0)) - F.pad(arg[:,0,:,:-1],(0,1,0,0)) # -(dx-)
        ret[:,0] += (F.pad(arg[:,1,:-1,:],(0,0,1,0)) - F.pad(arg[:,1,:-1,:],(0,0,0,1))) # -(dy-)
        return ret/self.h

    def check_adjointness(self, check_it, dim, dtype):
        print('\nAdjointness Check D:')
        for _ in range(check_it):
            u = torch.randn(1,1,dim,dim).type(dtype)
            w = torch.randn(1,2,dim,dim).type(dtype)
            lhs = (self.apply(u)*w).sum()
            rhs = (self.apply_adjoint(w)*u).sum()
            print('\tAjointness holds: %s (abs. diff. %.12f)' %(torch.allclose(lhs,rhs),torch.abs(lhs-rhs)))

class Eps: 
    def __init__(self,h=1.):
        # symmetric finite forward difference operator
        self.h = h

    def apply(self, arg):
        # (B,2,M1,M2) -> (B,3,M1,M2)
        B,_,M1,M2 = arg.shape
        ret = torch.zeros((B,3,M1,M2), dtype=arg.dtype, device=arg.device)

        # Dx arg1
        ret[:,0,:,:-1] += arg[:,0,:,1:] - arg[:,0,:,:-1]

        # Dy arg2
        ret[:,1,:-1,:] += arg[:,1,1:,:] - arg[:,1,:-1,:]

        # (Dy arg1 + Dx arg2)/2
        ret[:,2,:-1,:] += 0.5*(arg[:,0,1:,:] - arg[:,0,:-1,:])
        ret[:,2,:,:-1] += 0.5*(arg[:,1,:,1:] - arg[:,1,:,:-1])

        return ret/self.h

    def apply_adjoint(self, arg):
        # (B,3,M1,M2) -> (B,2,M1,M2)
        B,_,M1,M2 = arg.shape
        ret = torch.zeros((B,2,M1,M2), dtype=arg.dtype, device=arg.device)

        # Dx* arg1 + 0.5 Dy* arg3
        ret[:,0] = F.pad(arg[:,0,:,:-1],(1,0,0,0)) - F.pad(arg[:,0,:,:-1],(0,1,0,0))
        ret[:,0] += 0.5*(F.pad(arg[:,2,:-1,:],(0,0,1,0)) - F.pad(arg[:,2,:-1,:],(0,0,0,1)))

        # Dy* arg2 + 0.5 Dx* arg3
        ret[:,1] = (F.pad(arg[:,1,:-1,:],(0,0,1,0)) - F.pad(arg[:,1,:-1,:],(0,0,0,1)))
        ret[:,1] += 0.5*(F.pad(arg[:,2,:,:-1],(1,0,0,0)) - F.pad(arg[:,2,:,:-1],(0,1,0,0)))
        
        return ret/self.h

    def check_adjointness(self, check_it, dim, dtype):
        print('\nAdjointness Check Eps:')
        for _ in range(check_it):
            u = torch.randn(1,2,dim,dim).type(dtype)
            w = torch.randn(1,3,dim,dim).type(dtype)
            lhs = (self.apply(u)*w).sum()
            rhs = (self.apply_adjoint(w)*u).sum()
            print('\tAjointness holds: %s (abs. diff. %.12f)' %(torch.allclose(lhs,rhs),torch.abs(lhs-rhs)))

class D2:
    def __init__(self,h=1.):
        # finite forward difference operator (second order)
        self.h = h

    def apply(self, arg):
        # (B,1,M1,M2) -> (B,3,M1,M2)
        return Eps(self.h).apply(D(self.h).apply(arg))

    def apply_adjoint(self, arg):
        # (B,3,M1,M2) -> (B,1,M1,M2)
        return D(self.h).apply_adjoint(Eps(self.h).apply_adjoint(arg))

    def check_adjoitness(self, check_it, dim, dtype):
        print('\nAdjointness Check D2:')
        for _ in range(check_it):
            u = torch.randn(1,1,dim,dim).type(dtype)
            w = torch.randn(1,3,dim,dim).type(dtype)
            lhs = (self.apply(u)*w).sum()
            rhs = (self.apply_adjoint(w)*u).sum()
            print('\tAjointness holds: %s (abs. diff. %.12f)' %(torch.allclose(lhs,rhs),torch.abs(lhs-rhs)))

class prox_l2:
    def __init__(self, f, h=1.):
        self.f = f
        self.h = h
        
    def apply(self, arg, tau):
        # (B,1,M1,M2) -> (B,1,M1,M2)
        return (arg + tau*self.h*self.f)/(1 + tau*self.h)
    
    def apply_star(self, arg, tau):
        return (arg - tau*self.f)/(1 + tau)

class prox_l12: 
    def __init__(self, alpha, eps, d, n, h=1.):
        self.alpha = alpha
        if not isinstance(alpha,float): # for this case, expand alpha to enable broadcasting 
            self.alpha = self.alpha[:,None,None,None,None]
        self.eps = eps
        self.d = d # dimension of vector, d_p = 2, d_q = 3 
        self.n = n # number of interpolation positions in pixel
        self.h = h
    
    def norm(self, arg):
        if self.d == 2:
            return torch.sqrt((arg**2+self.eps).sum(1, keepdims=True))
        elif self.d == 3:
            return torch.sqrt(arg[:,0]**2 + arg[:,1]**2 + 2*arg[:,2]**2+self.eps)[:,None]

    def apply(self, arg, tau): 
        # (B,d*n,M1,M2) -> (B,d*n,M1,M2)
        B, _, M1, M2 = arg.shape
        arg = arg.reshape(B,self.d,self.n,M1,M2)
        denom = torch.clamp(self.norm(arg)/(tau*self.alpha*self.h**2), min=1.0)
        fact = 1.0 - 1.0/(denom + self.eps)
        return (arg*fact).reshape(B,-1,M1,M2)

    def apply_star(self, arg, tau):
        B, _, M1, M2 = arg.shape
        arg = arg.reshape(B,self.d,self.n,M1,M2)
        arg_norm = self.norm(arg)
        return (arg/torch.clamp(arg_norm/self.alpha, min=1.)).reshape(B,-1,M1,M2)

if __name__ == '__main__':
    h = 1.0 # h-scaling
    dtype = torch.float32
    check_it = 10
    dim = 100 
    
    # first- and second-order finite difference operators
    D(h).check_adjointness(check_it=check_it,dim=dim,dtype=dtype)
    Eps(h).check_adjointness(check_it=check_it,dim=dim,dtype=dtype)
    D2(h).check_adjoitness(check_it=check_it,dim=dim,dtype=dtype)