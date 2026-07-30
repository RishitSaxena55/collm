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
from models.lora import PEFTLoRA, apply_openclip_lora
from circo_utils import CIRCODataset, compute_metrics

# --- Datasets ---
def get_image_path(image_dir, img_id):
    path_png = os.path.join(image_dir, f"{img_id}.png")
    if os.path.exists(path_png):
        return path_png
    return os.path.join(image_dir, f"{img_id}.jpg")

class TargetDataset(Dataset):
    def __init__(self, target_ids, image_dir, transform):
        self.target_paths = [get_image_path(image_dir, tid) for tid in target_ids]
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
        
        ref_path = get_image_path(self.image_dir, ref_id)
        target_path = get_image_path(self.image_dir, target_id)
        
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

def run_circo_evaluation(config, vision_encoder, llm, adapter, device, transform):
    vision_encoder.eval()
    llm.eval()
    adapter.eval()
    
    data_path = config['data']['circo_dataset_dir']
    batch_size = config['training']['batch_size']
    
    print("\nInitializing CIRCO Datasets...")
    dataset_classic = CIRCODataset(data_path, split='val', mode='classic', preprocess=transform)
    dataset_relative = CIRCODataset(data_path, split='val', mode='relative', preprocess=transform)
    
    target_loader = DataLoader(dataset_classic, batch_size=batch_size, shuffle=False)
    query_loader = DataLoader(dataset_relative, batch_size=batch_size, shuffle=False)
    
    # 1. Target Embeddings
    print("Extracting Target Corpus Embeddings...")
    target_features = []
    target_ids = []
    
    with torch.no_grad():
        for batch in tqdm(target_loader):
            imgs = batch['img'].to(device)
            ids = batch['img_id'] # these are strings
            feats = vision_encoder(imgs)
            feats = F.normalize(feats, dim=-1)
            target_features.append(feats)
            target_ids.extend(ids)
            
    target_features = torch.cat(target_features, dim=0) # [N, dim]
    
    # 2. Query Embeddings
    print("Extracting Query Embeddings...")
    predictions_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(query_loader):
            ref_imgs = batch['reference_img'].to(device)
            caps = batch['relative_caption']
            q_ids = batch['query_id'] # list of str
            
            ref_feats = vision_encoder(ref_imgs)
            adapted_feats = adapter(ref_feats)
            
            query_feats = llm(adapted_feats, caps)
            query_feats = F.normalize(query_feats, dim=-1)
            
            # 3. Similarity and Ranking for this batch
            sims = query_feats @ target_features.T # [batch_size, N]
            topk_vals, topk_indices = sims.topk(50, dim=-1)
            
            for i, q_id in enumerate(q_ids):
                top_targets = [int(target_ids[idx]) for idx in topk_indices[i].cpu().tolist()]
                predictions_dict[int(q_id)] = top_targets
                
    # 4. Save and Compute Metrics
    import json
    from pathlib import Path
    with open("circo_val_predictions.json", "w") as f:
        json.dump(predictions_dict, f, indent=4)
        
    print("\nPredictions saved to circo_val_predictions.json. Computing official CIRCO metrics...")
    map_atk, recall_atk, semantic_map_at10 = compute_metrics(Path(data_path), predictions_dict, ranks=[5, 10, 25, 50])
    
    print("\nmAP@k metrics")
    for rank in [5, 10, 25, 50]:
        print(f"mAP@{rank}: {map_atk[rank] * 100:.2f}")

    print("\nRecall@k metrics")
    for rank in [5, 10, 25, 50]:
        print(f"Recall@{rank}: {recall_atk[rank] * 100:.2f}")

    print("\nSemantic mAP@10 metrics")
    for aspect, map_at10 in semantic_map_at10.items():
        print(f"Semantic mAP@10 for aspect '{aspect}': {map_at10 * 100:.2f}")

# --- Standalone Evaluation Loop ---
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional: Path to specific checkpoint.pt file to load (overrides default)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    v_lora_cfg = config['lora']['vision']
    vision_model_name = config['model']['vision_encoder_name']
    
    if vision_model_name.startswith("coca"):
        from models.vision_encoder import CoCaVisionEncoder
        vision_encoder = CoCaVisionEncoder(freeze=True).to(device)
        vision_dim = 768 # CoCa ViT-L-14 outputs 768
        if v_lora_cfg.get('enable', False):
            vision_encoder = apply_openclip_lora(
                vision_encoder, 
                r=v_lora_cfg['r'], 
                alpha=v_lora_cfg['alpha'], 
                target_modules=v_lora_cfg['target_modules']
            ).to(device)
    elif vision_model_name.startswith("open_clip:"):
        from models.vision_encoder import OpenCLIPVisionEncoder
        model_name, pretrained = vision_model_name.replace("open_clip:", "").split(",")
        vision_encoder = OpenCLIPVisionEncoder(model_name=model_name, pretrained=pretrained, freeze=True).to(device)
        vision_dim = vision_encoder.model.ln_pre.weight.shape[0] if hasattr(vision_encoder.model, "ln_pre") else 768
        if v_lora_cfg.get('enable', False):
            vision_encoder = apply_openclip_lora(
                vision_encoder, 
                r=v_lora_cfg['r'], 
                alpha=v_lora_cfg['alpha'], 
                target_modules=v_lora_cfg['target_modules']
            ).to(device)
    elif vision_model_name.startswith("blip:"):
        from models.vision_encoder import BLIPVisionEncoder
        model_name = vision_model_name.replace("blip:", "")
        if v_lora_cfg.get('enable', False):
            vision_lora = PEFTLoRA(r=v_lora_cfg['r'], lora_alpha=v_lora_cfg['alpha'], target_modules=v_lora_cfg['target_modules'], lora_dropout=v_lora_cfg['dropout'])
        else:
            vision_lora = None
        vision_encoder = BLIPVisionEncoder(model_name=model_name, freeze=True, lora_adapter=vision_lora).to(device)
        vision_dim = vision_encoder.model.config.hidden_size
    else:
        if v_lora_cfg.get('enable', False):
            vision_lora = PEFTLoRA(r=v_lora_cfg['r'], lora_alpha=v_lora_cfg['alpha'], target_modules=v_lora_cfg['target_modules'], lora_dropout=v_lora_cfg['dropout'])
        else:
            vision_lora = None
        vision_encoder = CLIPVisionEncoder(model_name=vision_model_name, freeze=True, lora_adapter=vision_lora).to(device)
        vision_dim = vision_encoder.model.config.hidden_size
    
    l_lora = config['lora']['llm']
    lora = PEFTLoRA(r=l_lora['r'], lora_alpha=l_lora['alpha'], target_modules=l_lora['target_modules'], lora_dropout=l_lora['dropout'])
    llm = SFREmbeddingLLM(vision_dim=vision_dim, freeze_llm=False, lora_adapter=lora).to(device)
    
    llm_dim = llm.llm.config.hidden_size
    
    adapter = ImageAdapter(vision_dim=vision_dim, llm_dim=llm_dim).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
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
    
    eval_dataset = config['data'].get('eval_dataset', 'fiq')
    if eval_dataset == 'circo':
        run_circo_evaluation(config, vision_encoder, llm, adapter, device, transform)
    else:
        run_evaluation(config, vision_encoder, llm, adapter, device, transform)

if __name__ == "__main__":
    evaluate()
