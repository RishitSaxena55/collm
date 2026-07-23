import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BatchSynthesizer(nn.Module):
    def __init__(self, vision_encoder, alpha=0.5, synthesis_prob=0.75):
        """
        Args:
            vision_encoder: The pre-trained, frozen vision model (e.g., CLIP) mapped to the GPU.
            alpha: Slerp interpolation weight.
            synthesis_prob: Probability to synthesize the text modification.
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.alpha = alpha
        self.synthesis_prob = synthesis_prob
        
        # Ensure vision encoder is frozen and in eval mode
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 15 templates from the paper's supplementary material
        self.templates = [
            "show {w_i} instead of {w_j}",
            "{w_i} instead of {w_j}",
            "show {w_i} rather than {w_j}",
            "{w_i} rather than {w_j}",
            "rather than {w_j}, show {w_i}",
            "rather than {w_j}, {w_i}",
            "instead of {w_j}, {w_i}",
            "{w_j}, changed to {w_i}",
            "not {w_j}, but {w_i}",
            "show {w_i}, not {w_j}",
            "{w_j} is missing, {w_i}",
            "{w_i}, and {w_j} is missing",
            "remove {w_j}, add {w_i}",
            "add {w_i}, remove {w_j}",
            "{w_j} become {w_i}"
        ]

    def forward(self, v_prime_batch, captions):
        """
        Args:
            v_prime_batch (Tensor): Augmented images batch (B, C, H, W), already on GPU.
            captions (list of str): Raw captions (len=B).
        Returns:
            h_star (Tensor): Synthesized visual embeddings.
            w_star (list of str): Synthesized modification text strings.
        """
        # 1. Extract Embeddings for augmented images
        with torch.no_grad():
            h_prime = self.vision_encoder(v_prime_batch)
            # L2 Normalize for cosine similarity computation
            h_prime_norm = F.normalize(h_prime, p=2, dim=-1)
            
        # 2. Find Nearest Neighbors within the batch
        # Compute cosine similarity matrix
        sim = torch.mm(h_prime_norm, h_prime_norm.t())
        
        # Mask diagonal (self-similarity) with -inf to avoid picking the exact same image
        sim.fill_diagonal_(float('-inf'))
        
        # Find index of nearest neighbor j for each item i
        j_indices = sim.argmax(dim=1)
        
        # 3. Synthesize Embeddings (Slerp)
        # Slerp formula: h* = (sin(a*theta)/sin(theta)) * h_i + (sin((1-a)*theta)/sin(theta)) * h_j
        h_prime_j = h_prime[j_indices]
        
        # Compute theta (angle between h'_i and h'_j)
        cos_theta = sim.max(dim=1).values
        # Clamp to avoid numerical issues (NaNs) with acos
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta).unsqueeze(1) # shape: (B, 1)
        
        sin_theta = torch.sin(theta)
        # Avoid division by zero if vectors are somehow identical
        sin_theta = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)
        
        coeff_i = torch.sin(self.alpha * theta) / sin_theta
        coeff_j = torch.sin((1.0 - self.alpha) * theta) / sin_theta
        
        # Slerp output (the synthesized reference image embedding)
        h_star = coeff_i * h_prime + coeff_j * h_prime_j
        
        # 4. Synthesize Text
        w_star = []
        for i in range(len(captions)):
            j = j_indices[i].item()
            w_i = captions[i]
            w_j = captions[j]
            
            # Apply synthesis with specified probability
            if random.random() < self.synthesis_prob:
                template = random.choice(self.templates)
                w_star_i = template.format(w_i=w_i, w_j=w_j)
            else:
                w_star_i = w_i
                
            w_star.append(w_star_i)
            
        return h_star, w_star
