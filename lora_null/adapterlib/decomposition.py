import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

class CorDA_adapter(nn.Module):
    def __init__(self, adapter_U, adapter_S, adapter_V, weight_residual, bias=None,sigma_fuse='UV') -> None:
        super().__init__()
        U, S, V = adapter_U, adapter_S, adapter_V     ## U: (m,r), V: (n,r), n==in_size, m==out_size
        rank = V.size(1)
        self.weight_residual = nn.Parameter(torch.zeros(U.size(0), V.size(0)).to(adapter_U.device))       # (m , n)
        self.weight_residual.data = weight_residual
        self.weight_residual.requires_grad = False
        
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)   ## r -> m
        
        if bias is not None:
            self.ALinear.bias.data = bias
        #self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)    --- > this is a bug of the original ASVD code, but can run normally
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)    ## n -> r

        if sigma_fuse == 'UV':
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == 'U':
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == 'V':
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y) + F.linear(inp, self.weight_residual)

        return y  


class CorDA_adapter2(nn.Module):
    def __init__(self, adapter_U, adapter_S, adapter_V, adapter_U2, weight_residual, bias=None,sigma_fuse='UV') -> None:
        super().__init__()
        U, S, V = adapter_U, adapter_S, adapter_V     ## U: (m,r), V: (n,r), n==in_size, m==out_size
        U2 = adapter_U2
        rank = V.size(1)
        self.weight_residual = nn.Parameter(torch.zeros(U.size(0), V.size(0)).to(adapter_U.device))       # (m , n)
        self.weight_residual.data = weight_residual
        self.weight_residual.requires_grad = False
        
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)   ## r -> m
        
        if bias is not None:
            self.ALinear.bias.data = bias
        #self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)    --- > this is a bug of the original ASVD code, but can run normally
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)    ## n -> r
        self.PBLinear = nn.Linear(adapter_U2.size(0), adapter_U2.size(1), bias = False)
        self.PALinear = nn.Linear(adapter_U2.size(1), adapter_U2.size(0), bias = False)
        self.PBLinear.weight.data = U2.t().contiguous()
        self.PALinear.weight.data = U2.contiguous()
        if sigma_fuse == 'UV':
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == 'U':
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == 'V':
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

    def forward(self, inp):
        # compute USV^Tx + b
        inp2 = inp - self.PALinear(self.PBLinear(inp))
        y = self.BLinear(inp2)
        y = self.ALinear(y) + F.linear(inp, self.weight_residual)

        return y   

    #@staticmethod
def full_decompose(
    linear: nn.Linear,  
    act_aware=False,   
    cov_aware=False,    
    proj_aware=False,   
    proj_x_aware=False,
    ori_proj_x_aware=False,
    proj_x_t_aware=False,
    cov_aware_wo_inv=False,
    alpha=1,            
    r=None,   ## lowest eigens to discard 
):
    rank = min(linear.in_features, linear.out_features)  
    w = linear.weight.data.float()  

    if act_aware:
        scaling_diag_matrix = 1  
        if hasattr(linear, "scaling_diag_matrix"):
            # print("WARNING: scaling_diag_matrix is used")
            scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
            # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
        if hasattr(linear, "fisher_info"):
            scaling_diag_matrix *= linear.fisher_info**alpha
            # scaling_diag_matrix *= linear.fisher_info**1
        # if not (scaling_diag_matrix == scaling_diag_matrix).all():
        #     breakpoint()
        scaling_diag_matrix += 1e-6  # avoid zero division
        w = w * scaling_diag_matrix.view(1, -1) 

    elif cov_aware:
        assert hasattr(linear, "covariance_matrix")
        covariance_matrix = linear.covariance_matrix.float() 
        damp = 0.01  
        while True:
            compensate = torch.diag(torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device)
                * torch.mean(torch.diag(covariance_matrix)) * damp)  
            fix_covariance_matrix = covariance_matrix + compensate 
            cov_inv = torch.linalg.inv(fix_covariance_matrix) 
            print("damp:", damp)
 
            inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).to(covariance_matrix.device))

            print("inv error:", inv_error)
            if inv_error.data < 0.05:   
                break
            else:
                damp = damp * 2
        w = w @ fix_covariance_matrix   

    try:
        U, S, V = torch.linalg.svd(w, full_matrices=False)  
        V = V.transpose(0, 1)   
    except:
        raise Exception("fsvd failed for {linear}")

    if act_aware:
        V = V / scaling_diag_matrix.view(-1, 1)  
    elif cov_aware:
        V = (V.t() @ cov_inv).transpose(0, 1)   

    if linear.bias is not None:
        bias = linear.bias.data  
    else:
        bias = None  

    if (S!=S).any():
        print("nan in S")
    if (U!=U).any():
        print("nan in U")
    if (V!=V).any():
        print("nan in V")
    if r is None:
        w_new = U @ torch.diag(S) @ V.transpose(0,1)  
    else:
        w_new = U[:, :rank-r] @ torch.diag(S[:rank-r]) @ V.transpose(0,1)[:rank-r,:] 
    w_new=w_new.to(linear.weight.dtype) 
    linear.weight.data = w_new  

    #@staticmethod
def decompose_to_adapter(
    linear: nn.Linear,
    act_aware=False,
    cov_aware=False,
    singular_aware = False,
    alpha=1,
    sigma_fuse="UV",
    r=16,
    first_eigen = False,
):
    rank = min(linear.in_features, linear.out_features)

    pretrained_w = linear.weight.data.float()#.cpu()
    if act_aware:
        scaling_diag_matrix = 1  # avoid zero division
        if hasattr(linear, "scaling_diag_matrix"):
            # print("WARNING: scaling_diag_matrix is used")
            scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
            # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
        if hasattr(linear, "fisher_info"):
            scaling_diag_matrix *= linear.fisher_info**alpha
            # scaling_diag_matrix *= linear.fisher_info**1
        # if not (scaling_diag_matrix == scaling_diag_matrix).all():
        #     breakpoint()
        scaling_diag_matrix += 1e-6  # avoid zero division
        w = pretrained_w * scaling_diag_matrix.view(1, -1)#.cpu()
    elif cov_aware:
        assert hasattr(linear, "covariance_matrix")
        covariance_matrix = linear.covariance_matrix.float()#.cpu()
        damp = 0.01
        while True:
            compensate = torch.diag(torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device)
                * torch.mean(torch.diag(covariance_matrix)) * damp)
            fix_covariance_matrix = covariance_matrix + compensate
            cov_inv = torch.linalg.inv(fix_covariance_matrix)
            print("damp:", damp)
            inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).to(covariance_matrix.device))
            print("inv error:", inv_error)
            if inv_error.data < 0.05:
                break
            else:
                damp = damp * 2
        w = pretrained_w @ fix_covariance_matrix   ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim

    try:
        if act_aware or cov_aware:
            #U, S, V = torch.svd_lowrank(w, q=rank)
            U, S, V = torch.linalg.svd(w, full_matrices=False)
            V = V.transpose(0, 1)
        else:
            #U, S, V = torch.svd_lowrank(pretrained_w, q=rank)
            U, S, V = torch.linalg.svd(pretrained_w, full_matrices=False)
            V = V.transpose(0, 1)
    except:
        raise Exception(f"svd failed for {linear}")

    if act_aware:
        V = V / scaling_diag_matrix.view(-1, 1)#.cpu()
    elif cov_aware:
        V = (V.t() @ cov_inv).transpose(0, 1)

    if linear.bias is not None:
        bias = linear.bias.data
    else:
        bias = None


    ## Use the last r principle components
    if not first_eigen:
        U = U[:,-r:]   ## m, r
        S = S[-r:]     ## r
        V = V[:,-r:]   ## n, r
    ## Use the first r principle components following PiSSA !!!
    elif first_eigen:
        U = U[:,:r]   ## m, r
        S = S[:r]     ## r
        V = V[:,:r]   ## n, r       
    ######################################

    weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0,1)  ## m,n

    if torch.isnan(weight_residual).any() or torch.isinf(weight_residual).any():
        raise Exception("nan or inf in weight_residual")

    linear_with_adapter = CorDA_adapter(U, S, V, weight_residual, bias, sigma_fuse)
    linear_with_adapter.to(linear.weight.dtype)#.cuda()
    linear_with_adapter.to(linear.weight.device)
    linear_with_adapter.weight_residual = linear_with_adapter.weight_residual.to(linear.weight.dtype)
    assert not torch.isnan(linear_with_adapter.weight_residual).any()
    assert not torch.isinf(linear_with_adapter.weight_residual).any()

    del pretrained_w, U, S, V, weight_residual, linear
    torch.cuda.empty_cache()

    return linear_with_adapter


def build_model(model, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    
    my_layers_keys = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            my_layers_keys.append(name)

    print('--- model before svd ----')
    print(model)
    #for layername, ratio in tqdm(my_layers_min_ratio.items()):
    for layername in tqdm(my_layers_keys):
        ###
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        ###      -----------------      #####
        with torch.no_grad():
        ## for full decompose 
            if args.mode == "full_decompose":
                full_decompose(
                    raw_linear,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    proj_aware = args.proj_aware,
                    proj_x_aware = args.proj_x_aware,
                    ori_proj_x_aware = args.proj_aware,
                    cov_aware_wo_inv = args.cov_aware_wo_inv,
                    proj_x_t_aware= args.proj_x_t_aware,
                    r=args.r
                )
            if args.mode == "build_adapters":
        ## for decompose to adapters
                if "lm_head" in layername:
                   continue
                linear_with_adapter = decompose_to_adapter(
                    raw_linear,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    r=args.r,
                    first_eigen = args.first_eigen,
                )
                delattr(info["father"], info["name"])
                if args.cov_aware: delattr(raw_linear, "covariance_matrix")
                if args.act_aware: delattr(raw_linear, "scaling_diag_matrix")
                setattr(info["father"], info["name"], linear_with_adapter)
                del module_dict[layername], linear_info[raw_linear]
                del raw_linear, info, 
                torch.cuda.empty_cache()
    print('--- model after svd ----')    
    print(model)


    #@staticmethod
def decompose_to_adapter2(
    linear: nn.Linear,
    act_aware=False,
    cov_aware=False,
    singular_aware = False,
    singular_aware_2 = False,
    alpha=1,
    sigma_fuse="UV",
    r=16,
    first_eigen = False,
):
    rank = min(linear.in_features, linear.out_features)

    pretrained_w = linear.weight.data.float()#.cpu()
    if act_aware:
        scaling_diag_matrix = 1  # avoid zero division
        if hasattr(linear, "scaling_diag_matrix"):
            # print("WARNING: scaling_diag_matrix is used")
            scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
            # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
        if hasattr(linear, "fisher_info"):
            scaling_diag_matrix *= linear.fisher_info**alpha
            # scaling_diag_matrix *= linear.fisher_info**1
        # if not (scaling_diag_matrix == scaling_diag_matrix).all():
        #     breakpoint()
        scaling_diag_matrix += 1e-6  # avoid zero division
        w = pretrained_w * scaling_diag_matrix.view(1, -1)#.cpu()
    elif cov_aware:
        assert hasattr(linear, "covariance_matrix")
        covariance_matrix = linear.covariance_matrix.float()#.cpu()
        damp = 0.01
        while True:
            compensate = torch.diag(torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device)
                * torch.mean(torch.diag(covariance_matrix)) * damp)
            fix_covariance_matrix = covariance_matrix + compensate
            cov_inv = torch.linalg.inv(fix_covariance_matrix)
            print("damp:", damp)
            inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).to(covariance_matrix.device))
            print("inv error:", inv_error)
            if inv_error.data < 0.05:
                break
            else:
                damp = damp * 2
        w = pretrained_w @ fix_covariance_matrix   ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim
    else:
        w = pretrained_w



    try:
        if act_aware or cov_aware:
            #U, S, V = torch.svd_lowrank(w, q=rank)
            U, S, V = torch.linalg.svd(w, full_matrices=False)
            V = V.transpose(0, 1)
        elif singular_aware:
            pass
        elif singular_aware_2:
            pass
        else:
            #U, S, V = torch.svd_lowrank(pretrained_w, q=rank)
            U, S, V = torch.linalg.svd(pretrained_w, full_matrices=False)
            V = V.transpose(0, 1)
    except:
        raise Exception(f"svd failed for {linear}")

    if act_aware:
        V = V / scaling_diag_matrix.view(-1, 1)#.cpu()
    elif cov_aware:
        V = (V.t() @ cov_inv).transpose(0, 1)

    if linear.bias is not None:
        bias = linear.bias.data
    else:
        bias = None

    # nan or inf check
    #if (S!=S).any():
    # print(111111111,singular_aware)
    # print(222222222,singular_aware_2)
    if not singular_aware and not singular_aware_2 and ( torch.isnan(S).any() or torch.isinf(S).any()):
        #print("nan in S")
        raise Exception("nan or inf in S")
    #if (U!=U).any():
    if not singular_aware and not singular_aware_2 and ( torch.isnan(U).any() or torch.isinf(U).any()):    
        #print("nan in U")
        raise Exception("nan or inf in U")
    #if (V!=V).any():
    if not singular_aware and not singular_aware_2 and ( torch.isnan(V).any() or torch.isinf(V).any()):    
        #print("nan in V")
        raise Exception("nan or inf in V")

    ## Use the last r principle components
    if not singular_aware and not singular_aware_2 and not first_eigen:
        U = U[:,-r:]   ## m, r
        S = S[-r:]     ## r
        V = V[:,-r:]   ## n, r
    ## Use the first r principle components following PiSSA !!!
    elif not singular_aware and not singular_aware_2 and first_eigen:
        U = U[:,:r]   ## m, r
        S = S[:r]     ## r
        V = V[:,:r]   ## n, r       
    ######################################

    if not singular_aware and not singular_aware_2:
        weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0,1)  ## m,n


    #weight_residual = weight_residual.to(linear.weight.dtype)#.cpu()
    #U = U.to(linear.weight.dtype)#.cpu()
    #S = S.to(linear.weight.dtype)#.cpu()
    #V = V.to(linear.weight.dtype)#.cpu()
    if singular_aware:
        
        covariance_matrix = linear.covariance_matrix.float()
        U_,S_, V_ = torch.linalg.svd(covariance_matrix)
        print((S_>0.1).sum())
        if first_eigen:
            U_min_K = U_[:,:r]
        else:
            U_min_K = U_[:,-r:]
            temp = pretrained_w @ (U_min_K @ U_min_K.t())
            weight_residual = pretrained_w - temp
            U, S, V = torch.svd(temp)
            U, S, V = U[:,:r], S[:r], V[:,:r]
            #V = U_min_K
            #S = torch.ones(r).to(V.dtype).to(V.device)
            #U = torch.zeros(weight_residual.shape[0],r).to(V.dtype).to(V.device)
            print(U.shape, S.shape, V.shape)
        '''
        if singluar_aware_freezeB:
            covariance_matrix = linear.covariance_matrix.float()
            U_,S_, V_ = torch.linalg.svd(covariance_matrix)
            print((S_>0.1).sum())
            if first_eigen:
                U_min_K = U_[:,:r]
            else:
                U_min_K = U_[:,-r:]
            temp = pretrained_w @ (U_min_K @ U_min_K.t())
            weight_residual = pretrained_w - temp
            U, S, V = torch.svd(temp)
            U, S, V = U[:,:r], S[:r], V[:,:r]
            #V = U_min_K
            #S = torch.ones(r).to(V.dtype).to(V.device)
            #U = torch.zeros(weight_residual.shape[0],r).to(V.dtype).to(V.device)
            print(U.shape, S.shape, V.shape)
        '''
    if singular_aware_2:
        covariance_matrix = linear.covariance_matrix.float()
        U_,S_, V_ = torch.linalg.svd(covariance_matrix)
        print((S_>0.1).sum())
        if first_eigen:
            U_min_K = U_[:,:r]
        else:
            U_min_K = U_[:,-r:]
        temp = pretrained_w @ (U_min_K @ U_min_K.t())
        weight_residual = pretrained_w
        U, S, V = torch.svd(temp)
        U, S, V = U[:,:r], S[:r], V[:,:r]
        V = U_min_K
        U.zero_()
        S.fill_(1.)

        print(U.shape, S.shape, V.shape)

        '''
        covariance_matrix = linear.covariance_matrix.float()
        U_,S_, V_ = torch.linalg.svd(covariance_matrix)
        E  = torch.sqrt(torch.diag(S_))

        temp = pretrained_w @ (U_ @ E)
        del U_,S_, V_ 
        try:
            E_inv = torch.linalg.inv(E)
            del E
        except Exception as e:
            print("Warning: scaling_diag_matrix is not full rank!")
            E += 1e-6 * torch.eye(E.shape[0]).to(E.device)
            E_inv = torch.linalg.inv(E)
            del E
        
        U, S, V = torch.linalg.svd(temp, full_matrices=True)
        V = V.transpose(0, 1)
        V = (V.t() @ E_inv).transpose(0, 1)
        if first_eigen:
            U = U[:,:r]   ## m, r
            S = S[:r]     ## r
            V = V[:,:r]   ## n, r   
        else:
            U = U[:,-r:]   ## m, r
            S = S[-r:]     ## r
            V = V[:,-r:]   ## n, r   
        ######################################
        weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0,1)  ## m,n
        '''
        
    if torch.isnan(weight_residual).any() or torch.isinf(weight_residual).any():
        raise Exception("nan or inf in weight_residual")
    
    linear_with_adapter = CorDA_adapter(U, S, V, weight_residual, bias, sigma_fuse)
    linear_with_adapter.to(linear.weight.dtype)#.cuda()
    linear_with_adapter.to(linear.weight.device)
    linear_with_adapter.weight_residual = linear_with_adapter.weight_residual.to(linear.weight.dtype)
    assert not torch.isnan(linear_with_adapter.weight_residual).any()
    assert not torch.isinf(linear_with_adapter.weight_residual).any()

    del pretrained_w, U, S, V, weight_residual, linear
    torch.cuda.empty_cache()

    return linear_with_adapter


def decompose_to_adapter3(
    linear: nn.Linear,
    act_aware=False,
    cov_aware=False,
    singular_aware = False,
    twin_aware = False,
    alpha=1,
    sigma_fuse="UV",
    r=16,
    first_eigen = False,
):
    rank = min(linear.in_features, linear.out_features)

    pretrained_w = linear.weight.data.float()#.cpu()
    if act_aware:
        scaling_diag_matrix = 1  # avoid zero division
        if hasattr(linear, "scaling_diag_matrix"):
            # print("WARNING: scaling_diag_matrix is used")
            scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
            # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
        if hasattr(linear, "fisher_info"):
            scaling_diag_matrix *= linear.fisher_info**alpha
            # scaling_diag_matrix *= linear.fisher_info**1
        # if not (scaling_diag_matrix == scaling_diag_matrix).all():
        #     breakpoint()
        scaling_diag_matrix += 1e-6  # avoid zero division
        w = pretrained_w * scaling_diag_matrix.view(1, -1)#.cpu()
    elif cov_aware:
        assert hasattr(linear, "covariance_matrix")
        covariance_matrix = linear.covariance_matrix.float()#.cpu()
        damp = 0.01
        while True:
            compensate = torch.diag(torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device)
                * torch.mean(torch.diag(covariance_matrix)) * damp)
            fix_covariance_matrix = covariance_matrix + compensate
            cov_inv = torch.linalg.inv(fix_covariance_matrix)
            print("damp:", damp)
            inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).to(covariance_matrix.device))
            print("inv error:", inv_error)
            if inv_error.data < 0.05:
                break
            else:
                damp = damp * 2
        w = pretrained_w @ fix_covariance_matrix   ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim


    try:
        if act_aware or cov_aware:
            #U, S, V = torch.svd_lowrank(w, q=rank)
            U, S, V = torch.linalg.svd(w, full_matrices=False)
            V = V.transpose(0, 1)
        elif singular_aware:
            pass
        else:
            #U, S, V = torch.svd_lowrank(pretrained_w, q=rank)
            U, S, V = torch.linalg.svd(pretrained_w, full_matrices=False)
            V = V.transpose(0, 1)
    except:
        raise Exception(f"svd failed for {linear}")

    if act_aware:
        V = V / scaling_diag_matrix.view(-1, 1)#.cpu()
    elif cov_aware:
        V = (V.t() @ cov_inv).transpose(0, 1)

    if linear.bias is not None:
        bias = linear.bias.data
    else:
        bias = None

    # nan or inf check
    #if (S!=S).any():
    print(111111111,singular_aware)
    if not singular_aware and ( torch.isnan(S).any() or torch.isinf(S).any()):
        #print("nan in S")
        raise Exception("nan or inf in S")
    #if (U!=U).any():
    if not singular_aware and ( torch.isnan(U).any() or torch.isinf(U).any()):    
        #print("nan in U")
        raise Exception("nan or inf in U")
    #if (V!=V).any():
    if not singular_aware and( torch.isnan(V).any() or torch.isinf(V).any()):    
        #print("nan in V")
        raise Exception("nan or inf in V")

    ## Use the last r principle components
    if not singular_aware and not first_eigen:
        U = U[:,-r:]   ## m, r
        S = S[-r:]     ## r
        V = V[:,-r:]   ## n, r
    ## Use the first r principle components following PiSSA !!!
    elif not singular_aware and first_eigen:
        U = U[:,:r]   ## m, r
        S = S[:r]     ## r
        V = V[:,:r]   ## n, r       
    ######################################

    if not singular_aware:
        weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0,1)  ## m,n


    if singular_aware:
        covariance_matrix = linear.covariance_matrix.float()
        U_,S_, V_ = torch.linalg.svd(covariance_matrix)
        if first_eigen:
            U_min_K = U_[:,:r]
        else:
            U_min_K = U_[:,-r:]
        temp = pretrained_w @ (U_min_K @ U_min_K.t())
        weight_residual = pretrained_w - temp
        U, S, V = torch.svd(temp)
        U, S, V = U[:,:r], S[:r], V[:,:r]
        del U_, S_, V_
    if twin_aware:
        covariance_matrix = linear.covariance_matrix.float()
        covariance_matrix2 = linear.covariance_matrix2.float()
        U_,S_, V_ = torch.linalg.svd(covariance_matrix)
        U2_,S2_, V2_ = torch.linalg.svd(covariance_matrix2)

        U2, S2, V2 = U2_[:,:r],S2_[:r], V2_[:,:r] 
        P = torch.eye(U2.size(0), dtype=U2.dtype, device=U2.device) - (U2 @ U2.t())
        #U3_, S3_, V3_ = torch.linalg.svd(P)
        #U3, S3, V3 = U3_[:,:r],S3_[:r], V3_[:,:r] 
        if first_eigen:
            U_min_K = U_[:,:r]
        else:
            U_min_K = U_[:,-r:]
        temp = pretrained_w @  (U_min_K @ U_min_K.t())

        weight_residual = pretrained_w - temp @ P
        U, S, V = torch.svd(temp)
        U, S, V = U[:,:r], S[:r], V[:,:r]
        del U_, S_, V_, U2_, S2_, V2_, S2, V2,  P

    if torch.isnan(weight_residual).any() or torch.isinf(weight_residual).any():
        raise Exception("nan or inf in weight_residual")
    if twin_aware:
        linear_with_adapter = CorDA_adapter2(U, S, V, U2, weight_residual, bias, sigma_fuse)
    else:
        linear_with_adapter = CorDA_adapter(U, S, V, weight_residual, bias, sigma_fuse)
    linear_with_adapter.to(linear.weight.dtype)#.cuda()
    linear_with_adapter.to(linear.weight.device)
    linear_with_adapter.weight_residual = linear_with_adapter.weight_residual.to(linear.weight.dtype)
    assert not torch.isnan(linear_with_adapter.weight_residual).any()
    assert not torch.isinf(linear_with_adapter.weight_residual).any()

    del pretrained_w, U, S, V, weight_residual, linear
    if twin_aware:
        del U2
    torch.cuda.empty_cache()

    return linear_with_adapter

def build_model2_mllm(model, args):

    module_dict = {name: module for name, module in model.model.layers.named_modules()}
    full_name_dict = {module: name for name, module in model.model.layers.named_modules()}
    linear_info = {}

    modules = [model.model.layers]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    
    my_layers_keys = []

    for name, module in model.model.layers.named_modules():
        if isinstance(module, nn.Linear):
            my_layers_keys.append(name)
    print("The Linear modules need to be changed.")
    print(my_layers_keys)
    print('--- model before svd ----')
    print(model)
    #for layername, ratio in tqdm(my_layers_min_ratio.items()):
    for layername in tqdm(my_layers_keys):
        ###
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        ###      -----------------      #####
        with torch.no_grad():
        ## for full decompose 
            if args.mode == "full_decompose":
                full_decompose(
                    raw_linear,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    proj_aware = args.proj_aware,
                    proj_x_aware = args.proj_x_aware,
                    ori_proj_x_aware = args.proj_aware,
                    cov_aware_wo_inv = args.cov_aware_wo_inv,
                    proj_x_t_aware= args.proj_x_t_aware,
                    r=args.r
                )
            if args.mode == "build_adapters":
        ## for decompose to adapters
                if "lm_head" in layername:
                   continue
                linear_with_adapter = decompose_to_adapter2(
                    raw_linear,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    singular_aware = args.singular_aware,
                    singular_aware_2 = args.singular_aware_2,
                    r=args.r,
                    first_eigen = args.first_eigen,
                )
                delattr(info["father"], info["name"])
                if args.cov_aware: delattr(raw_linear, "covariance_matrix")
                if args.act_aware: delattr(raw_linear, "scaling_diag_matrix")
                if args.singular_aware: delattr(raw_linear, "covariance_matrix")
                setattr(info["father"], info["name"], linear_with_adapter)
                del module_dict[layername], linear_info[raw_linear]
                del raw_linear, info, 
                torch.cuda.empty_cache()
    print('--- model after svd ----')    
    print(model)
    for name, param in model.model.layers.named_parameters():
        print(f"{name}: {param.shape}")

def build_model2(model, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    
    my_layers_keys = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            my_layers_keys.append(name)

    print('--- model before svd ----')
    print(model)
    #for layername, ratio in tqdm(my_layers_min_ratio.items()):
    for layername in tqdm(my_layers_keys):
        ###
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        ###      -----------------      #####
        with torch.no_grad():
        ## for full decompose 
            if args.mode == "full_decompose":
                full_decompose(
                    raw_linear,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    proj_aware = args.proj_aware,
                    proj_x_aware = args.proj_x_aware,
                    ori_proj_x_aware = args.proj_aware,
                    cov_aware_wo_inv = args.cov_aware_wo_inv,
                    proj_x_t_aware= args.proj_x_t_aware,
                    r=args.r
                )
            if args.mode == "build_adapters":
        ## for decompose to adapters
                if "lm_head" in layername:
                   continue
                linear_with_adapter = decompose_to_adapter2(
                    raw_linear,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    singular_aware = args.singular_aware,
                    singular_aware_2 = args.singular_aware_2,
                    r=args.r,
                    first_eigen = args.first_eigen,
                )
                delattr(info["father"], info["name"])
                if args.cov_aware: delattr(raw_linear, "covariance_matrix")
                if args.act_aware: delattr(raw_linear, "scaling_diag_matrix")
                if args.singular_aware: delattr(raw_linear, "covariance_matrix")
                setattr(info["father"], info["name"], linear_with_adapter)
                del module_dict[layername], linear_info[raw_linear]
                del raw_linear, info, 
                torch.cuda.empty_cache()
    print('--- model after svd ----')    
    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")


def build_model3(model, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    
    my_layers_keys = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            my_layers_keys.append(name)

    print('--- model before svd ----')
    print(model)
    #for layername, ratio in tqdm(my_layers_min_ratio.items()):
    for layername in tqdm(my_layers_keys):
        ###
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        ###      -----------------      #####
        with torch.no_grad():
        ## for full decompose 
            if args.mode == "full_decompose":
                full_decompose(
                    raw_linear,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    proj_aware = args.proj_aware,
                    proj_x_aware = args.proj_x_aware,
                    ori_proj_x_aware = args.proj_aware,
                    cov_aware_wo_inv = args.cov_aware_wo_inv,
                    proj_x_t_aware= args.proj_x_t_aware,
                    r=args.r
                )
            if args.mode == "build_adapters":
        ## for decompose to adapters
                if "lm_head" in layername:
                   continue
                linear_with_adapter = decompose_to_adapter3(
                    raw_linear,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    singular_aware = args.singular_aware,
                    twin_aware = args.twin_aware,
                    r=args.r,
                    first_eigen = args.first_eigen,
                )
                delattr(info["father"], info["name"])
                if args.cov_aware: delattr(raw_linear, "covariance_matrix")
                if args.act_aware: delattr(raw_linear, "scaling_diag_matrix")
                if args.singular_aware: delattr(raw_linear, "covariance_matrix")
                if args.twin_aware:
                    delattr(raw_linear, "covariance_matrix")
                    delattr(raw_linear, "covariance_matrix2")
                setattr(info["father"], info["name"], linear_with_adapter)
                del module_dict[layername], linear_info[raw_linear]
                del raw_linear, info, 
                torch.cuda.empty_cache()
    print('--- model after svd ----')    
    print(model)