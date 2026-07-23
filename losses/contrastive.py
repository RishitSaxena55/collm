import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CoLLMLoss(nn.Module):
    def __init__(self, init_tau=0.07):
        """
        Args:
            init_tau (float): Initial temperature parameter for InfoNCE.
        """
        super().__init__()
        # Learnable temperature parameter (stored as log_tau for numerical stability during optimization)
        # CLIP initializes logit_scale to np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_tau))
        
    def info_nce(self, q, k):
        """
        Computes the bidirectional InfoNCE loss between queries and targets.
        
        Args:
            q (Tensor): Query embeddings of shape (B, D)
            k (Tensor): Target embeddings of shape (B, D)
        Returns:
            Tensor: Scalar loss value.
        """
        device = q.device
        batch_size = q.size(0)
        
        # 1. L2 Normalize the embeddings
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # 2. Compute the scaled cosine similarity matrix
        # logit_scale is clamped to prevent exploding gradients (standard practice, max ~100)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits_per_q = logit_scale * torch.mm(q_norm, k_norm.t())
        logits_per_k = logits_per_q.t()
        
        # 3. Ground truth labels (the diagonal elements are the positive pairs)
        labels = torch.arange(batch_size, dtype=torch.long, device=device)
        
        # 4. Bidirectional Cross Entropy
        loss_q = F.cross_entropy(logits_per_q, labels)
        loss_k = F.cross_entropy(logits_per_k, labels)
        
        return (loss_q + loss_k) / 2.0

    def forward(self, c_composed, z, c_v=None, c_w=None):
        """
        Computes the total CoLLM loss as described in Eq 6 of the paper.
        For Stage 2 (Fine-tuning), c_v and c_w can be omitted to only compute the composed loss.
        
        Args:
            c_composed (Tensor): Multimodal composed query embeddings (B, D)
            z (Tensor): Target image embeddings from the vision encoder (B, D)
            c_v (Tensor, optional): Image-only query embeddings (B, D)
            c_w (Tensor, optional): Text-only query embeddings (B, D)
            
        Returns:
            Tensor: The averaged scalar loss.
        """
        loss_c = self.info_nce(c_composed, z)
        
        if c_v is not None and c_w is not None:
            loss_v = self.info_nce(c_v, z)
            loss_w = self.info_nce(c_w, z)
            # Equation 6: L = 1/3 * (L_cl(c_v, z) + L_cl(c_w, z) + L_cl(c, z))
            return (loss_v + loss_w + loss_c) / 3.0
            
        return loss_c
