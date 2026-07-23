import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer

class BaseLLM(nn.Module, ABC):
    def __init__(self, vision_dim, freeze_llm=True):
        """
        Abstract base class for the LLM component.
        
        Args:
            vision_dim: Dimension of the vision encoder (for the final projection).
            freeze_llm: If True, freezes the LLM weights.
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.freeze_llm = freeze_llm
        
        # The instruction template from the paper (Supp 7.2)
        self.template = (
            "Instruct: Find the image that matches the query.\n"
            "Query:\n"
            "Image: <IMAGE_TOKEN>\n"
            "Text: {text}"
        )
        
    def _apply_freezing(self, model):
        """Helper method to freeze the LLM parameters."""
        if self.freeze_llm:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
    @abstractmethod
    def forward(self, visual_embeds, text_list, modality="composed"):
        """
        Args:
            visual_embeds (Tensor): Tensor from ImageAdapter of shape (Batch, llm_dim).
                                    Can be None if modality is 'text_only'.
            text_list (list of str): List of synthesized text strings.
                                     Can be None or empty strings if modality is 'image_only'.
            modality (str): One of "composed", "image_only", "text_only".
                            Determines how the prompt is formatted.
        Returns:
            Tensor: The final composed query embedding of shape (Batch, vision_dim)
        """
        pass

class SFREmbeddingLLM(BaseLLM):
    def __init__(self, llm_model_name="Salesforce/SFR-Embedding-2_R", vision_dim=768, freeze_llm=True, lora_adapter=None, gradient_checkpointing=False):
        super().__init__(vision_dim=vision_dim, freeze_llm=freeze_llm)
        
        self.llm_model_name = llm_model_name
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Add special IMAGE token
        self.IMAGE_TOKEN = "<IMAGE_TOKEN>"
        num_added = self.tokenizer.add_tokens([self.IMAGE_TOKEN], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # Load LLM Base Model in bfloat16
        self.llm = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16)
        
        if gradient_checkpointing:
            # Disable KV cache (mandatory for checkpointing)
            self.llm.config.use_cache = False
            
            # Force input embeddings to require gradients (prevents LoRA autograd crashes)
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
                
            # Enable HF native checkpointing
            self.llm.gradient_checkpointing_enable()
            
        # Resize embeddings in case we added new tokens
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            
        self.llm_dim = self.llm.config.hidden_size
        
        # Final projection layer to map back to vision_dim for contrastive learning
        self.proj = nn.Linear(self.llm_dim, self.vision_dim, dtype=torch.bfloat16)
        
        # Apply LoRA or standard freezing
        if lora_adapter is not None:
            self.llm = lora_adapter.apply(self.llm)
        else:
            self._apply_freezing(self.llm)

    def _get_prompt(self, text, modality):
        if modality == "composed":
            return f"Instruct: Find the image that matches the query.\nQuery:\nImage: {self.IMAGE_TOKEN}\nText: {text}"
        elif modality == "image_only":
            return f"Instruct: Find the image that matches the query.\nQuery:\nImage: {self.IMAGE_TOKEN}"
        elif modality == "text_only":
            return f"Instruct: Find the image that matches the query.\nQuery:\nText: {text}"
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def forward(self, visual_embeds, text_list, modality="composed"):
        # Determine batch size and device based on provided inputs
        batch_size = visual_embeds.size(0) if visual_embeds is not None else len(text_list)
        device = visual_embeds.device if visual_embeds is not None else next(self.parameters()).device
        
        if text_list is None:
            text_list = [""] * batch_size

        # 1. Format the instruction strings
        prompts = [self._get_prompt(t, modality) for t in text_list]
        
        # 2. Tokenize
        tokenized = self.tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        
        # 3. Get raw token embeddings from the LLM
        # We MUST clone() here because we call enable_input_require_grads() on the LLM. 
        # Otherwise, the in-place visual embedding injection below crashes PyTorch autograd.
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        
        # 4. Inject visual embeddings if they exist in the prompt
        if modality in ["composed", "image_only"] and visual_embeds is not None:
            # Ensure visual_embeds matches the LLM dtype (bfloat16)
            visual_embeds = visual_embeds.to(inputs_embeds.dtype)
            
            for b_idx in range(batch_size):
                # Find where the <IMAGE_TOKEN> is in this specific sequence
                img_idx = (input_ids[b_idx] == self.image_token_id).nonzero(as_tuple=True)[0]
                if len(img_idx) > 0:
                    # Replace the placeholder embedding with the actual visual feature
                    inputs_embeds[b_idx, img_idx[0]] = visual_embeds[b_idx]
                
        # 5. Forward pass through the LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        # 6. Apply last_token_pool as used by SFR-Embedding
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            pooled_embeds = hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            pooled_embeds = hidden_states[torch.arange(batch_size, device=device), sequence_lengths]
            
        # 7. Final Projection to vision dimensional space
        # We cast back to float32 to ensure contrastive loss stability
        composed_query_embeds = self.proj(pooled_embeds).to(torch.float32)
        
        return composed_query_embeds
