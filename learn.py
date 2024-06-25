import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import argparse
import os
import json
import pickle
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

import torch
import torch.fft
import torch.nn.functional as F
import sys
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode 

try:
    from pad2d_op import pad2d, pad2dT
except ImportError:
    pass

import operators
import filters
import tgv_min

def parse_arguments(args): 
    parser = argparse.ArgumentParser('')
    parser.add_argument('--mode', type=str, default='denoise') 
    parser.add_argument('--alpha1', type=float, default=0.075) 
    parser.add_argument('--alpha0', type=float, default=2*0.075) 
    parser.add_argument('--eps_prox', type=float, default=1e-12) # small eps for prox (numerical stability)

    parser.add_argument('--pd_fact', type=int, default=1e0) # 1e1 for 0.05% noise, 1e0 for 0.1% noise
    parser.add_argument('--max_it', type=int, default=10000) # eval for more i.e. 10000 iterations
    
    parser.add_argument('--adjoint_mode', type=int, default=3) # V1: using autograd, V2: zero-padding, V3: CUDA op with more padding options
    parser.add_argument('--pad_mode', type=str, default='reflect') 
    parser.add_argument('--nK', type=int, default=4) 
    parser.add_argument('--nL', type=int, default=4) 
    parser.add_argument('--ker', type=int, default=3) 

    # piggyback settings
    parser.add_argument('--check_it', type=int, default=20) 
    parser.add_argument('--ad_it', type=int, default=100) 
    parser.add_argument('--etaK', type=float, default=1e-2) 
    parser.add_argument('--etaL', type=float, default=1e-2)
    parser.add_argument('--beta1', type=float, default=0.9) 
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps_adam', type=float, default=1e-8)  
    parser.add_argument('--save_filters', type=bool, default=True) 

    return parser.parse_args()

def proj_sum_equal(x):
    k,_,m,n = x.shape
    mu = x.sum()/(x.shape[0])
    x = x + 1/(m*n)*(mu-x.sum(axis=(-2,-1), keepdims=True))
    return x, mu

def loss_function(u,gt):
    K,_,M,N = u.shape
    loss = 0.5*((u-gt)**2).sum()/(K*M*N)
    return loss

def grad_loss_function(u,gt):
    K,_,M,N = u.shape
    grad_loss = (u-gt)/(K*M*N)
    return grad_loss

def pd_piggyback(u,ad_u,vK,ad_vK,vL,ad_vL,p,ad_p,u_gt,D2,K,L,Eps,prox_dt,prox_vK,prox_vL,cfg):
    # primal and dual step sizes according to block diagonal preconditioning
    delta_K = torch.abs(K.weights.reshape(3,cfg.nK,cfg.ker,cfg.ker).detach()).sum((-2,-1)).max(0)[0]
    delta_L = torch.abs(L.weights.reshape(2,cfg.nL,cfg.ker,cfg.ker).detach()).sum((-2,-1)).max(0)[0]
    theta = 1.0
    tau_u = cfg.pd_fact/(12./D2.h**2) 
    tau_vK = cfg.pd_fact/delta_K
    tau_vL = cfg.pd_fact/((3./Eps.h)*delta_L)
    sigma = 1/(4./D2.h**2 + delta_K.sum(0) + (2./Eps.h)*delta_L.sum(0))/cfg.pd_fact  

    # extend tau_vK and tau_vL dimensions along d=3/d=2 to enable quick element-wise multiplication 
    tau_vK = tau_vK[None,:].repeat(3,1).reshape(1,-1,1,1).to(u.device)
    tau_vL = tau_vL[None,:].repeat(2,1).reshape(1,-1,1,1).to(u.device)

    # pd algo
    for it in range(cfg.ad_it):
        # dual update, overrelaxation
        p_ = p.clone()
        p = p + sigma*(D2.apply(u) - K.apply_adjoint(vK) - Eps.apply(L.apply_adjoint(vL)))
        p_ = p + theta*(p-p_)

        # adjoint dual update, overrelaxation
        ad_p_ = ad_p.clone()
        ad_p = ad_p + sigma*(D2.apply(ad_u) - K.apply_adjoint(ad_vK) - Eps.apply(L.apply_adjoint(ad_vL)))
        ad_p_ = ad_p + theta*(ad_p-ad_p_)

        # primal update u
        u_tilde = u - tau_u*D2.apply_adjoint(p_)
        u_tilde.requires_grad_(True)
        u_hat = prox_dt.apply(u_tilde,tau_u)

        # compute gradient of loss
        grad_loss = grad_loss_function(u,u_gt)

        # adjoint primal update u
        ad_u_tilde = ad_u - tau_u*(D2.apply_adjoint(ad_p_) + grad_loss)
        ad_u = torch.autograd.grad(u_hat, u_tilde, ad_u_tilde)[0]
        u = u_hat.detach()

        # primal update vK
        vK_tilde = vK + tau_vK*K.apply(p_)
        vK_tilde.requires_grad_(True)
        vK_hat = prox_vK.apply(vK_tilde, tau_vK.reshape(1,3,cfg.nK,1,1)[:,0,...][:,None]) 
 
        # adjoint primal update vK
        ad_vK = ad_vK + tau_vK*K.apply(ad_p_)
        ad_vK = torch.autograd.grad(vK_hat, vK_tilde, ad_vK)[0]
        vK = vK_hat.detach()

        # primal update vL
        vL_tilde = vL + tau_vL*L.apply(Eps.apply_adjoint(p_))
        vL_tilde.requires_grad_(True)
        vL_hat = prox_vL.apply(vL_tilde, tau_vL.reshape(1,2,cfg.nL,1,1)[:,0,...][:,None])

        # adjoint primal update vL
        ad_vL = ad_vL + tau_vL*L.apply(Eps.apply_adjoint(ad_p_))
        ad_vL = torch.autograd.grad(vL_hat, vL_tilde, ad_vL)[0]
        vL = vL_hat.detach()

    return u, ad_u, vK, ad_vK, vL, ad_vL, p, ad_p

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
    K = filters.BilinearFilter(n=cfg.nK, d=3, ker=cfg.ker, pad_mode=cfg.pad_mode, adjoint_mode=cfg.adjoint_mode, dtype=dtype)
    L = filters.BilinearFilter(n=cfg.nL, d=2, ker=cfg.ker, pad_mode=cfg.pad_mode, adjoint_mode=cfg.adjoint_mode, dtype=dtype)

    # create all variables and adjoint variables
    u = torch.zeros_like(f)
    p = torch.zeros((B,3,M1,M2),dtype=dtype,device=device)
    vK = torch.zeros((B,3*cfg.nK,M1,M2),dtype=dtype,device=device) 
    vL = torch.zeros((B,2*cfg.nL,M1,M2),dtype=dtype,device=device) 

    ad_u = torch.zeros_like(f)
    ad_p = torch.zeros((B,3,M1,M2),dtype=dtype,device=device)
    ad_vK = torch.zeros((B,3*cfg.nK,M1,M2),dtype=dtype,device=device) 
    ad_vL = torch.zeros((B,2*cfg.nL,M1,M2),dtype=dtype,device=device) 

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

    # ADAM optimizer parameters
    m_K = torch.zeros_like(K.weights.clone())
    m_L = torch.zeros_like(L.weights.clone())
    v_K = torch.zeros_like(K.weights.clone())
    v_L = torch.zeros_like(L.weights.clone())

    # learning iterations
    loss = []
    for i in range(cfg.max_it):
        # piggyback algo
        u,ad_u,vK,ad_vK,vL,ad_vL,p,ad_p = pd_piggyback(u=u,ad_u=ad_u,vK=vK,ad_vK=ad_vK,vL=vL,ad_vL=ad_vL,p=p,ad_p=ad_p,u_gt=u_gt,D2=D2,K=K,L=L,Eps=Eps,prox_dt=prox_dt,prox_vK=prox_vK,prox_vL=prox_vL,cfg=cfg)

        # compute gradient wrt K and L
        K.weights.requires_grad_(True)
        L.weights.requires_grad_(True)
        sp = -(ad_vK*K.apply(p)).sum() - (ad_vL*L.apply(Eps.apply_adjoint(p))).sum() - (vK*K.apply(ad_p)).sum() - (vL*L.apply(Eps.apply_adjoint(ad_p))).sum()
        grad = torch.autograd.grad(sp, [K.weights, L.weights])
        grad_K, grad_L = grad[0], grad[1]

        # update biased first moment estimate
        m_K = m_K*cfg.beta1 + (1-cfg.beta1)*grad_K
        m_L = m_L*cfg.beta1 + (1-cfg.beta1)*grad_L

        # update biased second raw moment estimate
        v_K = v_K*cfg.beta2 + (1-cfg.beta2)*(grad_K**2)
        v_L = v_L*cfg.beta2 + (1-cfg.beta2)*(grad_L**2)

        # compute bias-corrected first moment estimate
        m_K_ = m_K/(1-cfg.beta1)
        m_L_ = m_L/(1-cfg.beta1)

        # compute bias-corrected second raw moment estimate
        v_K_ = v_K/(1-cfg.beta2)
        v_L_ = v_L/(1-cfg.beta2)
        
        # update + constraint on filters
        K.weights = K.weights.detach_()
        step_K = torch.max(torch.sqrt(v_K_).reshape(v_K_.shape[0],-1),dim=-1)[0][:,None,None,None]
        K.weights = K.weights - cfg.etaK*m_K_/(step_K+cfg.eps_adam)
        K.weights, mu_K = proj_sum_equal(K.weights)

        L.weights = L.weights.detach_()
        step_L = torch.max(torch.sqrt(v_L_).reshape(v_L_.shape[0],-1),dim=-1)[0][:,None,None,None]
        L.weights = L.weights - cfg.etaL*m_L_/(step_L+cfg.eps_adam)
        L.weights, mu_L = proj_sum_equal(L.weights)
      
        K.weights = K.weights.detach()
        L.weights = L.weights.detach()

        # logging
        loss.append(loss_function(u,u_gt).item())

        if i % cfg.check_it == 0:        
            print("Learn-TGV: it = ", i,
                    ", PSNR = ", "{:3.4f} dB".format(10.*torch.log10(1/((u-u_gt)**2).mean((-2,-1))).mean().item()),
                    ", mu_K = ", "{:.4f}".format(mu_K),
                    ", mu_L = ", "{:.4f}".format(mu_L),
                    ", loss = ", loss[-1],
                    end="\n")

    if cfg.save_filters:
        np.savez(os.getcwd() + '/learned_parameters.npz', K=K.weights.cpu().numpy(), L=L.weights.cpu().numpy(), mu_K=mu_K.item(), mu_L=mu_L.item())

    # quantitative evaluation
    f_mse, u_mse = ((f-u_gt)**2).mean((-2,-1)), ((u-u_gt)**2).mean((-2,-1))
    f_psnr, u_psnr = 10.*torch.log10(1/(((f-u_gt)**2).mean((-2,-1)))), 10.*torch.log10(1/(((u-u_gt)**2).mean((-2,-1))))

    fig, ax = plt.subplots(1,3,sharex=True,sharey=True)
    [a.axis('off') for a in ax.reshape(-1)]
    ax[0].imshow(f[0,0].cpu(),'gray'), ax[0].set_title('PSNR = %.4fdB' %(10*torch.log10(1/((f[0,0]-u_gt[0,0])**2).mean((-2,-1)))).item())
    ax[1].imshow(u[0,0].cpu(),'gray'), ax[1].set_title('PSNR = %.4fdB' %(10*torch.log10(1/((u[0,0]-u_gt[0,0])**2).mean((-2,-1)))).item())
    ax[2].imshow(u_gt[0,0].cpu(),'gray'), ax[2].set_title('groundtruth')
    plt.show()