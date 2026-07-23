import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageAdapter(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        """
        Maps the vision features into the semantic space of the Large Language Model.
        
        Args:
            vision_dim (int): Dimension of the visual features (e.g., 768 for CLIP-L/14)
            llm_dim (int): Embedding dimension of the chosen LLM
        """
        super().__init__()
        
        # A 2-layer MLP (Linear -> GELU -> Linear) as is commonly used in modern MLLMs (like LLaVA)
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, visual_features):
        """
        Args:
            visual_features (Tensor): Tensor of shape (Batch, vision_dim)
                                      typically the pooled output from the VisionEncoder.
        Returns:
            Tensor of shape (Batch, llm_dim) ready for the LLM.
        """
        return self.proj(visual_features)
