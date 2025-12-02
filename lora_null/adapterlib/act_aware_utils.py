import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

def calib_fisher_info(model, calib_loader, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    torch.save(all_fisher_info, cache_file)


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    # 确保缓存目录存在
    cache_dir = os.path.dirname(cache_file) or "."
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(
                    module.weight.device
                )
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)   ## input[0]: (1, 2048, dim); input[0].abs().mean(dim=-2): (1, dim); abs_mean: (dim)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    #count=0
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}  ## batch['input_ids']: (1, 2048)
        #print(count, ' batch data:', batch['input_ids'].size())
        #count+=1
        model(**batch)
        #print(model.model.layers[0].mlp.up_proj.scaling_diag_matrix)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    torch.save(all_scaling_diag_matrix, cache_file)


@torch.no_grad()
def calib_cov_distribution(
    model,
    calib_loader,
    use_cache=True,
    calib_dataset="wiki",
    calib_size=256,
    dataset_names=None,
    n_samples_per_dataset=None,
    benchmark_cache_file=None,
    seed=None,
):
    model_id = model.config._name_or_path
    cache_file = benchmark_cache_file if benchmark_cache_file else (
        f"cache/{model_id.replace('/','_')}_covariance_matrices_from_{calib_dataset}_size_{calib_size}_seed_{seed}.pt"
    )
    # 确保缓存目录存在
    cache_dir = os.path.dirname(cache_file) or "."
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_file) and use_cache:
        print(f"covariance cache file found: {cache_file}")
        all_covariance_matrix = torch.load(cache_file, map_location="cpu")
        # 只对LM部分做操作
        for name, module in model.model.layers.named_modules():
            if isinstance(module, nn.Linear):
                module.covariance_matrix = all_covariance_matrix[name].to(
                    module.weight.device
                )
        return
    model.eval()

    print(f"building covariance file: {cache_file}")
    print("new new new")
    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data   ## (2048, dim)

        input = input

        input = input/torch.max(input).abs()

        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")
        
        # 对于bs,seq,hidden_dim的input，计算其协方差矩阵时，先将bs和seq维度合并
        input_2d = input.view(-1, input.size(-1))  # (bs*seq, hidden_dim)
        covariance = input_2d.t().matmul(input_2d)
        del input_2d  
        
        # covariance = input.t().matmul(input)

        if torch.isnan(covariance).any():
            print("nan detected")
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            print("inf detected")
            raise Exception("inf in covariance, break")        
        module.covariance_matrix += covariance/256 
        del covariance, input
    # 只对LM部分做操作
    for name, module in model.model.layers.named_modules():
        if isinstance(module, nn.Linear):
            module.covariance_matrix = 0
            module.register_forward_hook(hook)
    
    for batch in tqdm(calib_loader):
        # batch = {k: v.to(model.device) for k,v in batch.items() if k!="image_size"}
        model(**batch)
        print("-"*10)
    
    #####################################
    # for batch in tqdm(calib_loader):
    #     input_ids, image_tensors, image_sizes = batch
    #     model(input_ids=input_ids.to(model.device), images=image_tensors.to(model.device), image_sizes=image_sizes, use_cache=True)
    #####################################

    all_covariance_matrix = {}
    # 只对LM部分做操作
    for name, module in model.model.layers.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            if torch.isnan(module.covariance_matrix).any():
                print("nan detected")
                raise Exception("nan in covariance")
            if torch.isinf(module.covariance_matrix).any():
                print("inf detected")
                raise Exception("inf in covariance")
            module.covariance_matrix = module.covariance_matrix     #/ 256
            all_covariance_matrix[name] = module.covariance_matrix
    
    torch.save(all_covariance_matrix, cache_file)  # this file would be large
    print("covariance matrices saved")




def calib_fisher_info(model, calib_loader, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    torch.save(all_fisher_info, cache_file)


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(
                    module.weight.device
                )
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)   ## input[0]: (1, 2048, dim); input[0].abs().mean(dim=-2): (1, dim); abs_mean: (dim)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    #count=0
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}  ## batch['input_ids']: (1, 2048)
        #print(count, ' batch data:', batch['input_ids'].size())
        #count+=1
        model(**batch)
        #print(model.model.layers[0].mlp.up_proj.scaling_diag_matrix)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    torch.save(all_scaling_diag_matrix, cache_file)


@torch.no_grad()
def calib_cov_distribution2(model, calib_loader, use_cache=True, calib_dataset="wiki", calib_size=256, seed=None):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_covariance_matrices_from_{calib_dataset}_size_{calib_size}_seed_{seed}.pt"
    )

    if os.path.exists(cache_file) and use_cache:
        print(f"covariance cache file found: {cache_file}")
        all_covariance_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.covariance_matrix2 = all_covariance_matrix[name].to(
                    module.weight.device
                )
        return

    model.eval()

    print(f"building covariance file: {cache_file}")
    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data   ## (2048, dim)

        input = input / torch.max(input).abs()

        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")
        
        covariance = input.t().matmul(input)

        if torch.isnan(covariance).any():
            print("nan detected")
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            print("inf detected")
            raise Exception("inf in covariance, break")        
        module.covariance_matrix2 += covariance/256 
        del covariance, input


    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.covariance_matrix2 = 0
            module.register_forward_hook(hook)
    
    
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k,v in batch.items()}
        model(**batch)

    all_covariance_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            if torch.isnan(module.covariance_matrix).any():
                print("nan detected")
                raise Exception("nan in covariance")
            if torch.isinf(module.covariance_matrix).any():
                print("inf detected")
                raise Exception("inf in covariance")
            module.covariance_matrix2 = module.covariance_matrix2     #/ 256
            all_covariance_matrix[name] = module.covariance_matrix2
    #torch.save(all_covariance_matrix, cache_file)  # this file would be large
    print("covariance matrices saved")





