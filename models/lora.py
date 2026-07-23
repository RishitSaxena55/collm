import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLoRA(ABC):
    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Applies LoRA to the given model.
        
        Args:
            model (nn.Module): The base PyTorch model to adapt.
            
        Returns:
            nn.Module: The model wrapped with LoRA adapters.
        """
        pass

class PEFTLoRA(BaseLoRA):
    def __init__(self, 
                 r: int = 16, 
                 lora_alpha: int = 16, 
                 target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"], 
                 lora_dropout: float = 0.1,
                 task_type: str = "FEATURE_EXTRACTION"):
        """
        Concrete implementation of LoRA using Hugging Face's PEFT library.
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.task_type = task_type
        
    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("Please install the 'peft' library to use PEFTLoRA: pip install peft")
            
        config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=self.task_type
        )
        
        # Apply LoRA using PEFT
        peft_model = get_peft_model(model, config)
        
        # Ensure gradients are required on the LoRA parameters
        peft_model.train()
        
        return peft_model

class OpenCLIPLoRAParametrization(nn.Module):
    def __init__(self, weight_shape, r=16, alpha=16, target_q=True, target_k=True, target_v=True):
        super().__init__()
        # weight_shape is typically (3 * dim, dim)
        out_features, in_features = weight_shape
        self.r = r
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        
        import math
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Build the mask for the 3 chunks (Q, K, V)
        # We assume out_features == 3 * dim
        dim = out_features // 3
        mask = torch.zeros(out_features, 1)
        if target_q:
            mask[0:dim, 0] = 1.0
        if target_k:
            mask[dim:2*dim, 0] = 1.0
        if target_v:
            mask[2*dim:3*dim, 0] = 1.0
            
        self.register_buffer("mask", mask)
        
    def forward(self, W):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return W + (delta * self.mask)

class StandardLoRAParametrization(nn.Module):
    def __init__(self, weight_shape, r=16, alpha=16):
        super().__init__()
        out_features, in_features = weight_shape
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        
        import math
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, W):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return W + delta

def apply_openclip_lora(model: nn.Module, r: int, alpha: int, target_modules: list):
    """
    Applies parametrization-based LoRA to an OpenCLIP model.
    """
    import torch.nn.utils.parametrize as parametrize
    
    target_q = "q_proj" in target_modules
    target_k = "k_proj" in target_modules
    target_v = "v_proj" in target_modules
    target_out = "out_proj" in target_modules
    
    for name, module in model.named_modules():
        # OpenCLIP Attention or PyTorch Native MultiheadAttention
        if module.__class__.__name__ in ["Attention", "MultiheadAttention"]:
            if target_q or target_k or target_v:
                # OpenCLIP fuses QKV into in_proj_weight
                if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                    # Freeze the base weight first
                    module.in_proj_weight.requires_grad = False
                    param_lora = OpenCLIPLoRAParametrization(
                        module.in_proj_weight.shape, r=r, alpha=alpha, 
                        target_q=target_q, target_k=target_k, target_v=target_v
                    )
                    parametrize.register_parametrization(module, "in_proj_weight", param_lora)
                    
            if target_out:
                if hasattr(module, "out_proj"):
                    # out_proj is an nn.Linear
                    module.out_proj.weight.requires_grad = False
                    std_lora = StandardLoRAParametrization(module.out_proj.weight.shape, r=r, alpha=alpha)
                    parametrize.register_parametrization(module.out_proj, "weight", std_lora)
                    
    return model
