import argparse
import json
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from models.vision_encoder import CLIPVisionEncoder
from models.llm import SFREmbeddingLLM
from models.adapter import ImageAdapter
from models.lora import PEFTLoRA

# --- Datasets ---
class TargetDataset(Dataset):
    def __init__(self, target_ids, image_dir, transform):
        self.target_paths = [os.path.join(image_dir, f"{tid}.jpg") for tid in target_ids]
        self.transform = transform
        
    def __len__(self):
        return len(self.target_paths)
        
    def __getitem__(self, idx):
        path = self.target_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (FileNotFoundError, OSError):
            img = Image.new('RGB', (224, 224))
        return self.transform(img), path

class FIQQueryDataset(Dataset):
    def __init__(self, json_path, image_dir, transform):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        ref_id = item.get("candidate")
        target_id = item.get("target")
        
        ref_path = os.path.join(self.image_dir, f"{ref_id}.jpg")
        target_path = os.path.join(self.image_dir, f"{target_id}.jpg")
        
        texts = item.get("captions", [])
        mod_text = " and ".join(texts) if isinstance(texts, list) else ""
            
        try:
            img = Image.open(ref_path).convert('RGB')
        except (FileNotFoundError, OSError):
            img = Image.new('RGB', (224, 224))
            
        return self.transform(img), mod_text, target_path

def run_evaluation(config, vision_encoder, llm, adapter, device, transform):
    vision_encoder.eval()
    llm.eval()
    adapter.eval()
    
    eval_dir = config['data']['eval_dataset_dir']
    eval_split = config['data']['eval_split']
    
    # Bug Fix: We must use the FIQ images directory for evaluation, 
    # not the config['data']['image_dir'] which points to LLaVA or MTCIR!
    image_dir = os.path.join(eval_dir, "images")
    
    batch_size = config['training']['batch_size']
    
    classes = ["dress", "shirt", "toptee"]
    aggregated_metrics = {'Recall@1': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'Recall@25': 0.0, 'Recall@50': 0.0}
    
    print(f"\nStarting Multi-Class FIQ Evaluation across {classes}...")
    
    for cls in classes:
        print(f"\n================ Evaluating Class: {cls.upper()} ================")
        
        target_file = os.path.join(eval_dir, "image_splits", f"split.{cls}.{eval_split}.json")
        query_file = os.path.join(eval_dir, "captions", f"cap.{cls}.{eval_split}.json")
        
        # 1. Extract Target Corpus
        print(f"Loading target corpus from {target_file}...")
        with open(target_file, 'r') as f:
            target_ids = json.load(f)
        print(f"Found {len(target_ids)} target images in the split.")
        
        target_dataset = TargetDataset(target_ids, image_dir, transform)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
        
        # Encode Targets
        print(f"[{cls.upper()}] Encoding Target Corpus...")
        target_embeddings = []
        target_paths_ordered = []
        with torch.no_grad():
            for imgs, paths in tqdm(target_loader):
                z_i = vision_encoder(imgs.to(device))
                z_i = F.normalize(z_i, p=2, dim=-1)
                target_embeddings.append(z_i.cpu())
                target_paths_ordered.extend(paths)
        target_embeddings = torch.cat(target_embeddings, dim=0) # [N, 1024]
        
        # 2. Encode Queries
        print(f"[{cls.upper()}] Encoding Queries...")
        query_dataset = FIQQueryDataset(query_file, image_dir, transform)
        query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
        
        query_embeddings = []
        gt_targets = []
        with torch.no_grad():
            for imgs, texts, targets in tqdm(query_loader):
                h_i = vision_encoder(imgs.to(device))
                adapted_vision = adapter(h_i)
                # LLM expects list of strings for text_queries
                c_i = llm(visual_embeds=adapted_vision, text_list=list(texts), modality="composed")
                c_i = F.normalize(c_i, p=2, dim=-1)
                query_embeddings.append(c_i.cpu())
                gt_targets.extend(targets)
        query_embeddings = torch.cat(query_embeddings, dim=0) # [M, 1024]
        
        # 3. Compute Similarities and Recall@K
        print(f"[{cls.upper()}] Computing Similarities and Recall metrics...")
        similarities = torch.matmul(query_embeddings, target_embeddings.T) # [M, N]
        
        ranks = []
        for i in range(len(gt_targets)):
            gt = gt_targets[i]
            sims = similarities[i].tolist()
            
            # Sort targets by similarity (descending)
            ranked_indices = sorted(range(len(sims)), key=lambda k: sims[k], reverse=True)
            ranked_paths = [target_paths_ordered[idx] for idx in ranked_indices]
            
            try:
                rank = ranked_paths.index(gt) + 1
                ranks.append(rank)
            except ValueError:
                pass # GT not in corpus
                
        if not ranks:
            print(f"No valid ranks found for {cls}.")
            continue
            
        # Calculate Recall for this class
        class_metrics = {
            "Recall@1": sum(1 for r in ranks if r <= 1) / len(ranks),
            "Recall@5": sum(1 for r in ranks if r <= 5) / len(ranks),
            "Recall@10": sum(1 for r in ranks if r <= 10) / len(ranks),
            "Recall@25": sum(1 for r in ranks if r <= 25) / len(ranks),
            "Recall@50": sum(1 for r in ranks if r <= 50) / len(ranks),
        }
        
        print(f"Results for {cls.upper()}:")
        for k, v in class_metrics.items():
            print(f"  {k}: {v * 100:.2f}%")
            aggregated_metrics[k] += v
            
    print("\n" + "="*30)
    print("      FIQ AVERAGED EVALUATION RESULTS")
    print("="*30)
    for k in aggregated_metrics.keys():
        aggregated_metrics[k] = (aggregated_metrics[k] / len(classes)) * 100
        print(f"{k:>10}: {aggregated_metrics[k]:.2f}%")
    print("="*30 + "\n")
    
    return aggregated_metrics

# --- Standalone Evaluation Loop ---
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    v_lora = config['lora']['vision']
    vision_lora = PEFTLoRA(r=v_lora['r'], lora_alpha=v_lora['alpha'], target_modules=v_lora['target_modules'], lora_dropout=v_lora['dropout'])
    vision_encoder = CLIPVisionEncoder(model_name=config['model']['vision_encoder_name'], freeze=True, lora_adapter=vision_lora).to(device)
    
    vision_dim = vision_encoder.model.config.hidden_size
    
    l_lora = config['lora']['llm']
    lora = PEFTLoRA(r=l_lora['r'], lora_alpha=l_lora['alpha'], target_modules=l_lora['target_modules'], lora_dropout=l_lora['dropout'])
    llm = SFREmbeddingLLM(vision_dim=vision_dim, freeze_llm=False, lora_adapter=lora).to(device)
    
    llm_dim = llm.llm.config.hidden_size
    
    adapter = ImageAdapter(vision_dim=vision_dim, llm_dim=llm_dim).to(device)
    
    # Standalone eval attempts to load last_checkpoint.pt from output directory
    ckpt_path = os.path.join(config['training']['output_dir'], "last_checkpoint.pt")
    print(f"Loading checkpoint from {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        vision_encoder.load_state_dict(checkpoint['vision_encoder'], strict=False)
        llm.load_state_dict(checkpoint['llm'], strict=False)
        adapter.load_state_dict(checkpoint['adapter'])
    except Exception as e:
        print(f"Failed to load checkpoint. Error: {e}")
        return
        
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    run_evaluation(config, vision_encoder, llm, adapter, device, transform)

if __name__ == "__main__":
    evaluate()
