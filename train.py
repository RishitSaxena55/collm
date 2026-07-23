import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.image_caption_dataset import ImageCaptionDataset, LLaVADataset
from data.triplet_dataset import MTCIRDataset, FIQDataset
from models.vision_encoder import CLIPVisionEncoder, CoCaVisionEncoder
from models.adapter import ImageAdapter
from utils.synthesis import BatchSynthesizer
from models.llm import SFREmbeddingLLM
from models.lora import PEFTLoRA, apply_openclip_lora
from losses.contrastive import CoLLMLoss
import argparse
import yaml
import os
import torchvision.transforms as T
from eval import run_evaluation
from accelerate import Accelerator
from utils.distributed import gather_embeddings

def count_params(module):
    if module is None:
        return 0, 0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def print_training_summary(vision_encoder, llm, adapter, config, dataloader, dataset_name, accelerator):
    print("="*60)
    print("="*16 + " TRAINING CONFIGURATION " + "="*18)
    print("="*60)
    
    stage = config['training']['stage']
    print(f"Stage:                {stage} ({'Pre-training' if stage == 1 else 'Fine-tuning'})")
    print(f"Dataset:              {dataset_name}")
    print(f"Epochs:               {config['training']['epochs']}")
    per_device_batch = config['training']['batch_size']
    global_batch = per_device_batch * accelerator.num_processes
    print(f"Per-Device Batch Size:{per_device_batch}")
    print(f"Global Batch Size:    {global_batch} ({per_device_batch} x {accelerator.num_processes} GPUs)")
    print(f"Steps per Epoch:      {len(dataloader)}")
    print(f"Learning Rate:        {config['training']['learning_rate']}")
    print(f"Mixed Precision:      {config.get('accelerate', {}).get('mixed_precision', 'no')}")
    print(f"Grad Checkpointing:   {'Enabled' if config['training'].get('gradient_checkpointing', False) else 'Disabled'}")
    print("-"*60)
    
    v_tot, v_trn = count_params(vision_encoder)
    l_tot, l_trn = count_params(llm)
    a_tot, a_trn = count_params(adapter)
    
    print("--- Vision Encoder ---")
    print(f"Name:                 {config['model']['vision_encoder_name']}")
    v_lora = config['lora']['vision']
    if v_lora.get('enable', False):
        print(f"LoRA:                 Enabled (r={v_lora['r']}, alpha={v_lora['alpha']}, targets={v_lora['target_modules']})")
    else:
        print("LoRA:                 Disabled")
    print(f"Total Params:         {v_tot:,}")
    print(f"Trainable Params:     {v_trn:,} ({v_trn/v_tot*100:.4f}%)" if v_tot > 0 else "Trainable Params:     0")
    print()
    
    print("--- Language Model (LLM) ---")
    print(f"Name:                 {config['model']['llm_name']}")
    l_lora = config['lora']['llm']
    print(f"LoRA:                 Enabled (r={l_lora['r']}, alpha={l_lora['alpha']}, targets={l_lora['target_modules']})")
    print(f"Total Params:         {l_tot:,}")
    print(f"Trainable Params:     {l_trn:,} ({l_trn/l_tot*100:.4f}%)" if l_tot > 0 else "Trainable Params:     0")
    print()
    
    print("--- Adapter ---")
    print("Name:                 2-layer MLP (Linear -> GELU -> Linear)")
    print(f"Total Params:         {a_tot:,}")
    print(f"Trainable Params:     {a_trn:,} ({a_trn/a_tot*100:.4f}%)" if a_tot > 0 else "Trainable Params:     0")
    print("-"*60)
    
    sys_tot = v_tot + l_tot + a_tot
    sys_trn = v_trn + l_trn + a_trn
    print("=== AGGREGATE TOTALS ===")
    print(f"System Total Params:      {sys_tot:,}")
    print(f"System Trainable Params:  {sys_trn:,} ({sys_trn/sys_tot*100:.4f}%)" if sys_tot > 0 else "System Trainable Params:  0")
    print("="*60)
    print()

def main():
    parser = argparse.ArgumentParser(description="Train CoLLM")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mixed_precision = config['training'].get('mixed_precision', 'no')
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Using device: {device} with mixed precision: {mixed_precision}")

    # 1. Data Loading
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    stage = config['training'].get('stage', 1)
    dataset_type = config['data'].get('dataset_type', 'llava')

    if stage == 1:
        dataset = LLaVADataset(config['data']['train_dataset_file'], image_dir=config['data']['image_dir'], target_transform=transform, aug_transform=transform)
    else:
        if dataset_type == "mtcir":
            dataset = MTCIRDataset(config['data']['train_dataset_file'], image_dir=config['data']['image_dir'], transform=transform)
        elif dataset_type == "fiq":
            dataset = FIQDataset(config['data']['train_dataset_file'], image_dir=config['data']['image_dir'], transform=transform)
        else:
            raise ValueError(f"Unknown dataset_type for stage 2: {dataset_type}")
            
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 2. Model Initialization
    # Instantiate the Vision Encoder
    v_lora_cfg = config['lora']['vision']
    vision_model_name = config['model']['vision_encoder_name']
    
    if vision_model_name.startswith("coca"):
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
    else:
        if v_lora_cfg.get('enable', False):
            vision_lora = PEFTLoRA(r=v_lora_cfg['r'], lora_alpha=v_lora_cfg['alpha'], target_modules=v_lora_cfg['target_modules'], lora_dropout=v_lora_cfg['dropout'])
        else:
            vision_lora = None
        vision_encoder = CLIPVisionEncoder(model_name=vision_model_name, freeze=True, lora_adapter=vision_lora).to(device)
        vision_dim = vision_encoder.model.config.hidden_size
    
    # Instantiate the LoRA Adapter and LLM
    l_lora = config['lora']['llm']
    lora = PEFTLoRA(r=l_lora['r'], lora_alpha=l_lora['alpha'], target_modules=l_lora['target_modules'], lora_dropout=l_lora['dropout'])
    llm = SFREmbeddingLLM(
        vision_dim=vision_dim, 
        freeze_llm=False, 
        lora_adapter=lora,
        gradient_checkpointing=config['training'].get('gradient_checkpointing', False)
    ).to(device)
    
    llm_dim = llm.llm.config.hidden_size
    
    adapter = ImageAdapter(vision_dim=vision_dim, llm_dim=llm_dim).to(device)
    
    if stage == 1:
        synthesizer = BatchSynthesizer(vision_encoder=vision_encoder).to(device)
    else:
        synthesizer = None
        
    criterion = CoLLMLoss(init_tau=0.07).to(device)

    # Optimizer (only trainable parts)
    all_modules = [vision_encoder, adapter, llm, criterion]
    trainable_params = []
    for module in all_modules:
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])
        
    optimizer = torch.optim.AdamW(trainable_params, lr=config['training']['learning_rate'])
    
    vision_encoder, llm, adapter, optimizer, dataloader, criterion = accelerator.prepare(
        vision_encoder, llm, adapter, optimizer, dataloader, criterion
    )

    # Print Summary
    if accelerator.is_main_process:
        # Determine dataset name for logging
        if stage == 1:
            dataset_name = config['data'].get('llava_file', 'LLaVA')
        elif dataset_type == "mtcir":
            dataset_name = config['data'].get('mtcir_dataset_name', 'MTCIR')
        else:
            dataset_name = "FashionIQ"
            
        print_training_summary(vision_encoder, llm, adapter, config, dataloader, dataset_name, accelerator)
        
    # Construct global run name
    base_name = config.get('wandb', {}).get('run_name', 'PTbb_coca_large')
    dataset_type_str = config['data'].get('dataset_type', 'llava')
    global_batch = config['training']['batch_size'] * accelerator.num_processes
    run_name = f"{base_name}_stage{stage}_{dataset_type_str}_b{global_batch}"
        
    # Set up output directory appended with run name
    output_dir = os.path.join(config['training']['output_dir'], run_name)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    eval_k = config['training']['eval_every_k_steps']
    
    global_step = 0
    best_recall_10 = 0.0
    start_epoch = 0
    steps_to_skip = 0
    
    resumed_successfully = False
    resume_checkpoint = config['training'].get('resume_from_checkpoint')
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            vision_encoder.load_state_dict(checkpoint['vision_encoder'], strict=False)
            llm.load_state_dict(checkpoint['llm'], strict=False)
            adapter.load_state_dict(checkpoint['adapter'])
            
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            global_step = checkpoint.get('global_step', 0)
            best_recall_10 = checkpoint.get('best_recall_10', 0.0)
            
            start_epoch = global_step // len(dataloader)
            steps_to_skip = global_step % len(dataloader)
            
            print(f"Resumed successfully. Fast-forwarding to Epoch {start_epoch+1}, Step {steps_to_skip}")
            resumed_successfully = True
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")

    # 3. Training Loop
    if accelerator.is_main_process:
        print("Starting training loop...")
        
    wandb_enabled = config.get('wandb', {}).get('enable', False)
    if wandb_enabled and accelerator.is_main_process:
        import wandb
        
        wandb.init(
            entity="navjak-carnegie-mellon-university",
            project="ret-collm",
            name=run_name,
            config=config
        )
        
    # Zero-shot evaluation before training
    if not resumed_successfully and global_step == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("\n--- Running Zero-Shot Evaluation before training ---")
            metrics = run_evaluation(config, accelerator.unwrap_model(vision_encoder), 
                                   accelerator.unwrap_model(llm), accelerator.unwrap_model(adapter), 
                                   device, transform)
            if wandb_enabled and metrics:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
            print("---------------------------------------------------\n")
        
    for epoch in range(start_epoch, config['training']['epochs']):
        for step, batch in enumerate(dataloader):
            if epoch == start_epoch and step < steps_to_skip:
                continue
                
            # Ensure models are in train mode
            vision_encoder.train()
            llm.train()
            adapter.train()
            
            optimizer.zero_grad()
            
            if stage == 1:
                v_i, v_prime_i, captions = batch
                v_i = v_i.to(device)
                v_prime_i = v_prime_i.to(device)
                
                with torch.no_grad():
                    z_i = vision_encoder(v_i)
                    h_star_i, w_star_i = synthesizer(v_prime_i, captions)
                    
                visual_embeds = adapter(h_star_i)
                c_composed, c_v, c_w = llm(visual_embeds=visual_embeds, text_list=w_star_i, modality="stage1", text_list_text_only=captions)
                
                # Gather embeddings globally for InfoNCE loss
                z_i = gather_embeddings(z_i)
                c_composed = gather_embeddings(c_composed)
                c_v = gather_embeddings(c_v)
                c_w = gather_embeddings(c_w)
                
                loss = criterion(c_composed=c_composed, z=z_i, c_v=c_v, c_w=c_w)
                
            else:
                v_ref, v_target, mod_text = batch
                v_ref = v_ref.to(device)
                v_target = v_target.to(device)
                
                with torch.no_grad():
                    z_i = vision_encoder(v_target)
                    h_i = vision_encoder(v_ref)
                
                visual_embeds = adapter(h_i)
                c_composed = llm(visual_embeds=visual_embeds, text_list=list(mod_text), modality="composed")
                
                # Gather embeddings globally for InfoNCE loss
                z_i = gather_embeddings(z_i)
                c_composed = gather_embeddings(c_composed)
                
                loss = criterion(c_composed=c_composed, z=z_i, c_v=None, c_w=None)

            accelerator.backward(loss)
            optimizer.step()
            
            global_step += 1
            
            if accelerator.is_main_process:
                print(f"Epoch [{epoch+1}/{config['training']['epochs']}] Step [{step+1}/{len(dataloader)}] (Global {global_step}) Loss: {loss.item():.4f}")
                if wandb_enabled:
                    wandb.log({
                        "train/loss_total": loss.item(),
                        "train/lr": optimizer.param_groups[0]['lr']
                    }, step=global_step)
            
            # G. In-Training Evaluation
            if global_step % eval_k == 0:
                accelerator.wait_for_everyone() # Ensure all GPUs catch up before evaluating
                if accelerator.is_main_process:
                    print(f"\n--- Running Evaluation at Global Step {global_step} ---")
                    metrics = run_evaluation(config, accelerator.unwrap_model(vision_encoder), 
                                           accelerator.unwrap_model(llm), accelerator.unwrap_model(adapter), 
                                           device, transform)
                    
                    # Create checkpoint dict
                    checkpoint = {
                        'vision_encoder': accelerator.unwrap_model(vision_encoder).state_dict(),
                        'llm': accelerator.unwrap_model(llm).state_dict(),
                        'adapter': accelerator.unwrap_model(adapter).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'global_step': global_step,
                        'best_recall_10': best_recall_10
                    }
                    
                    # Save last checkpoint
                    last_path = os.path.join(output_dir, "last_checkpoint.pt")
                    torch.save(checkpoint, last_path)
                    print(f"Saved {last_path}")
                    
                    # Check for best checkpoint
                    r10 = metrics.get('Recall@10', 0.0)
                    if r10 > best_recall_10:
                        best_recall_10 = r10
                        best_path = os.path.join(output_dir, "best_checkpoint.pt")
                        torch.save(checkpoint, best_path)
                        print(f"*** New Best Recall@10 ({r10 * 100:.2f}%)! Saved {best_path} ***")
                        
                    if wandb_enabled and metrics:
                        wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
                        
                    print("---------------------------------------------------\n")

    if accelerator.is_main_process:
        print("Training loop complete!")

if __name__ == "__main__":
    main()
