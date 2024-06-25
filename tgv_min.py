import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os

import torch
import torch.fft
import torch.nn.functional as F
import sys

try:
    from pad2d_op import pad2d, pad2dT
except ImportError:
    pass

import operators
import filters

def parse_arguments(args): 
    parser = argparse.ArgumentParser('')
    parser.add_argument('--mode', type=str, default='denoise') 
    parser.add_argument('--alpha1', type=float, default=0.075) 
    parser.add_argument('--alpha0', type=float, default=2*0.075) 
    parser.add_argument('--eps_prox', type=float, default=1e-12) # small eps for prox (numerical stability)

    parser.add_argument('--pd_fact', type=int, default=1e0) # 1e1 for 0.05% noise, 1e0 for 0.1% noise
    parser.add_argument('--max_it', type=int, default=10000) # eval for more i.e. 10000 iterations
    
    parser.add_argument('--adjoint_mode', type=int, default=1) # V1: using autograd, V2: zero-padding, V3: CUDA op with more padding options
    parser.add_argument('--pad_mode', type=str, default='reflect') 
    parser.add_argument('--use_ref', type=bool, default=False) 
    parser.add_argument('--nK', type=int, default=4) 
    parser.add_argument('--nL', type=int, default=4) 
    parser.add_argument('--ker', type=int, default=3) 

    # load learned filters
    parser.add_argument('--load_filters', type=bool, default=False) 

    return parser.parse_args()

def min(u,vK,vL,p,D2,K,L,Eps,prox_dt,prox_vK,prox_vL,pd_it,cfg,u_gt=None):
    # primal and dual stepsizes using diagonal block preconditioning 
    theta = 1.0
    tau_u = cfg.pd_fact/(12./D2.h**2)
    if cfg.use_ref == True:  # for reference TGV
        tau_vK = cfg.pd_fact/1.
        tau_vL = cfg.pd_fact/((3./Eps.h)*1)
        sigma = 1/(4./D2.h**2 + 1. + (2./Eps.h)*1.)/cfg.pd_fact
    else:
        delta_K = torch.abs(K.weights.reshape(3,cfg.nK,cfg.ker,cfg.ker).detach()).sum((-2,-1)).max(0)[0]
        delta_L = torch.abs(L.weights.reshape(2,cfg.nL,cfg.ker,cfg.ker).detach()).sum((-2,-1)).max(0)[0]
        tau_vK = cfg.pd_fact/delta_K
        tau_vL = cfg.pd_fact/((3./Eps.h)*delta_L)
        sigma = 1/(4./D2.h**2 + delta_K.sum(0) + (2./Eps.h)*delta_L.sum(0))/cfg.pd_fact
    
        # extend tau_vK and tau_vL dimensions along d=3/d=2 for element-wise multiplication 
        tau_vK = tau_vK[None,:].repeat(3,1).reshape(1,-1,1,1).to(u.device)
        tau_vL = tau_vL[None,:].repeat(2,1).reshape(1,-1,1,1).to(u.device)

    # primal-dual iterations
    for it in range(pd_it):
        p_ = p.clone()
        p = p + sigma*(D2.apply(u) - K.apply_adjoint(vK) - Eps.apply(L.apply_adjoint(vL)))
        p_= p + theta*(p-p_)

        u = prox_dt.apply(u - tau_u*D2.apply_adjoint(p_),tau_u)
        if cfg.use_ref == True:
            vK = prox_vK.apply(vK + tau_vK*K.apply(p_), tau_vK)  
            vL = prox_vL.apply(vL + tau_vL*L.apply(Eps.apply_adjoint(p_)), tau_vL) 
        else:
            vK = prox_vK.apply(vK + tau_vK*K.apply(p_), tau_vK.reshape(1,3,cfg.nK,1,1)[:,0,...][:,None]) 
            vL = prox_vL.apply(vL + tau_vL*L.apply(Eps.apply_adjoint(p_)), tau_vL.reshape(1,2,cfg.nL,1,1)[:,0,...][:,None])  

    return u, vK, vL, p

if __name__ == "__main__":
    cfg = parse_arguments(sys.argv)
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load data 
    u_gt = torch.from_numpy(cv2.imread(os.getcwd() + '/data/lena.png',0)/255.).unsqueeze(0).unsqueeze(0).type(dtype).to(device)
    u_gt = (u_gt-u_gt.min())/(u_gt.max()-u_gt.min())
    f = torch.from_numpy(np.load(os.getcwd() + '/data/lena_noisy.npy')).type(dtype).to(device) 
    B,_,M1,M2 = u_gt.shape

    # interpolation filters
    if cfg.use_ref:
        print('Careful: reference TGV selected, set nK=1 and nL=1!')
        K = filters.IdFilter()
        L = filters.IdFilter()
        cfg.nK, cfg.nL = 1, 1
    else:
        K = filters.BilinearFilter(n=cfg.nK, d=3, ker=cfg.ker, pad_mode=cfg.pad_mode, adjoint_mode=cfg.adjoint_mode, dtype=dtype)
        L = filters.BilinearFilter(n=cfg.nL, d=2, ker=cfg.ker, pad_mode=cfg.pad_mode, adjoint_mode=cfg.adjoint_mode, dtype=dtype)

        if cfg.load_filters:
            print('Using learned filters!')
            learned_params = np.load(os.getcwd() + '/learned_parameters.npz')
            K.weights = torch.from_numpy(learned_params['K']).to(device).type(dtype)
            L.weights = torch.from_numpy(learned_params['L']).to(device).type(dtype)

    # create all variables
    u = torch.zeros_like(f)
    p = torch.zeros((B,3,M1,M2),dtype=dtype,device=device)
    vK = torch.zeros((B,3*cfg.nK,M1,M2),dtype=dtype,device=device) 
    vL = torch.zeros((B,2*cfg.nL,M1,M2),dtype=dtype,device=device) 

    # operators 
    h = 1. # discretization 
    D2 = operators.D2(h=h)
    Eps = operators.Eps(h=h)

    if cfg.mode == 'denoise':
        prox_dt = operators.prox_l2(f)
    else: 
        raise ValueError('Not yet implemented!')
    prox_vL = operators.prox_l12(alpha=cfg.alpha1,eps=cfg.eps_prox,d=2,n=cfg.nL)
    prox_vK = operators.prox_l12(alpha=cfg.alpha0,eps=cfg.eps_prox,d=3,n=cfg.nK)

    # TGV regularized minimization problem
    u,vK,vL,p = min(u=u,vK=vK,vL=vL,p=p,D2=D2,K=K,L=L,Eps=Eps,prox_dt=prox_dt,prox_vK=prox_vK,prox_vL=prox_vL,pd_it=cfg.max_it,cfg=cfg,u_gt=u_gt)

    # quantitative evaluation
    f_mse, u_mse = ((f-u_gt)**2).mean((-2,-1)), ((u-u_gt)**2).mean((-2,-1))
    f_psnr, u_psnr = 10.*torch.log10(1/(((f-u_gt)**2).mean((-2,-1)))), 10.*torch.log10(1/(((u-u_gt)**2).mean((-2,-1))))

    fig, ax = plt.subplots(1,3,sharex=True,sharey=True)
    [a.axis('off') for a in ax.reshape(-1)]
    ax[0].imshow(f[0,0].cpu(),'gray'), ax[0].set_title('PSNR = %.4fdB' %(10*torch.log10(1/((f[0,0]-u_gt[0,0])**2).mean((-2,-1)))).item())
    ax[1].imshow(u[0,0].cpu(),'gray'), ax[1].set_title('PSNR = %.4fdB' %(10*torch.log10(1/((u[0,0]-u_gt[0,0])**2).mean((-2,-1)))).item())
    ax[2].imshow(u_gt[0,0].cpu(),'gray'), ax[2].set_title('groundtruth')
    plt.show()