from transformers import Qwen2_5_VLForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F
import torch

class CovSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.weight_residual = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_residual.requires_grad = False


    def forward(self, input):
        y = self.BLinear(input)
        y = self.ALinear(y) + F.linear(input, self.weight_residual)
        return y


from transformers import Qwen2_5_VLConfig


class CoVSVDQwen2_5_VLConfig(Qwen2_5_VLConfig):
    def __init__(self, lora_r=128, **kwargs):
        super().__init__(**kwargs)
        self.lora_r = lora_r


        

class CovSVDQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = CoVSVDQwen2_5_VLConfig

    def __init__(self, config: CoVSVDQwen2_5_VLConfig):
        super().__init__(config)
        
        self.lora_r = config.lora_r


   

        # # 就是这一行代码，它正在做您想做的事情！
        # config = CoVSVDQwen2_5_VLConfig.from_pretrained("/home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/Knowledge_Oriented3/qwen_kore_base_ckpt2")
        # 只对LM做修改
        # print(self.model)

        
        full_name_dict = {module: name for name, module in self.model.layers.named_modules()}
        linear_info = {}
        modules = [self.model.layers]
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

        # 只对LM做修改
        for name,module in self.model.layers.named_modules():
            if "lm_head" not in name and isinstance(module, nn.Linear):
                info=linear_info[module]
                new_layer=CovSVDLinear(module.in_features, module.out_features, self.lora_r, bias=module.bias is not None)
                setattr(info["father"], info["name"], new_layer)