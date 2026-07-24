import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import CLIPVisionModel, BlipForConditionalGeneration


class BaseVisionEncoder(nn.Module, ABC):
    def __init__(self, freeze=True):
        super().__init__()
        self.freeze = freeze
        
    def _apply_freezing(self):
        """Helper method to freeze or unfreeze the model parameters."""
        if self.freeze:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True
                
    @abstractmethod
    def forward(self, pixel_values):
        """
        Args:
            pixel_values (Tensor): Image tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, Hidden_Dim) representing the pooled 1D visual token.
        """
        pass

class BLIPVisionEncoder(BaseVisionEncoder):
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", freeze=True, lora_adapter=None):
        super().__init__(freeze)
        
        # Bug Fix: Salesforce/blip-image-captioning-large is a full multimodal model.
        # If we load it directly via BlipVisionModel, HF fails to map the vision_model.* keys
        # and randomly initializes the vision encoder. We must load the full model and extract it.
        full_model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model = full_model.vision_model
        
        if lora_adapter is not None:
            self.model = lora_adapter.apply(self.model)
        else:
            self._apply_freezing()
            
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.pooler_output

class CLIPVisionEncoder(BaseVisionEncoder):
    def __init__(self, model_name="openai/clip-vit-large-patch14", freeze=True, lora_adapter=None):
        super().__init__(freeze)
        self.model_name = model_name
        # Load the raw Vision Model without any specific task heads
        self.model = CLIPVisionModel.from_pretrained(model_name)
        
        if lora_adapter is not None:
            self.model = lora_adapter.apply(self.model)
        else:
            # Apply freezing exactly as specified by BaseVisionEncoder
            self._apply_freezing()
        
    def forward(self, pixel_values):
        # We just need the pooler_output (global feature from the [CLS] token)
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.pooler_output

class OpenCLIPVisionEncoder(BaseVisionEncoder):
    def __init__(self, model_name="ViT-L-14", pretrained="openai", freeze=True):
        super().__init__(freeze)
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.visual
        
        # Remove projection to match HuggingFace CLIPVisionModel behavior (returns raw 1024 dim instead of 768)
        if hasattr(self.model, 'proj'):
            self.model.proj = None
            
        self._apply_freezing()
        
    def forward(self, pixel_values):
        self.model.output_tokens = False
        pooled = self.model(pixel_values)
        return pooled

class CoCaVisionEncoder(BaseVisionEncoder):
    def __init__(self, freeze=True):
        super().__init__(freeze)
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('coca_ViT-L-14', pretrained='laion2b_s13b_b90k')
        self.model = model.visual
        self._apply_freezing()
        
    def forward(self, pixel_values):
        # OpenCLIP's VisionTransformer with output_tokens=False returns just the pooled tensor
        self.model.output_tokens = False
        pooled = self.model(pixel_values)
        return pooled
