import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from tqdm import tqdm

logger = logging.getLogger("CIRREval")

def evaluate_cirr_dataset(
    vision_encoder,
    llm,
    adapter,
    queries_loader,
    pool_loader,
    device,
    split: str = "val",
    save_dir: Optional[str] = None,
    exp_name: str = "cirr_eval"
) -> Dict[str, float]:
    """
    Evaluate CoLLM on CIRR dataset split.
    If split == "test1", generates submission JSON files without metric calculation.
    If split == "val" (or "dev"), computes global R@1,5,10,25,50 and subset R@1,2,3.
    """
    vision_encoder.eval()
    llm.eval()
    adapter.eval()

    # Step 1: Encode Gallery Pool
    pool_embs = []
    pool_ids = []
    
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            for batch in tqdm(pool_loader, desc=f"Encoding CIRR {split} gallery"):
                images = batch["target_images"].to(device)
                
                embeds = vision_encoder(images)
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
                
                pool_embs.append(embeds.cpu())
                pool_ids.extend(batch["target_id"])
                
    pool_embs = torch.cat(pool_embs, dim=0)
    pool_id_to_idx = {img_id: idx for idx, img_id in enumerate(pool_ids)}

    # Step 2: Encode Queries and gather metadata
    q_embs = []
    q_target_ids = []
    q_ref_ids = []
    q_subset_ids = []
    q_pairids = []

    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            for batch in tqdm(queries_loader, desc=f"Encoding CIRR {split} queries"):
                ref_imgs = batch["ref_images"].to(device)
                texts = batch["texts"]

                v_i = vision_encoder(ref_imgs)
                v_prime_i = adapter(v_i)
                emb = llm(visual_embeds=v_prime_i, text_list=texts, modality="composed")
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                q_embs.append(emb.cpu())
                q_target_ids.extend(batch["target_id"])
                q_ref_ids.extend(batch["ref_id"])
                q_subset_ids.extend(batch["subset_ids"])
                q_pairids.extend(batch["pairid"])

    q_embs = torch.cat(q_embs, dim=0)
    num_queries = len(q_pairids)

    # Step 3: Compute global similarities and mask reference image
    # Shape: (num_queries, num_gallery)
    global_scores = (q_embs.to(device, dtype=torch.float32) @ pool_embs.to(device, dtype=torch.float32).T)
    
    # Mask reference images
    for i, ref_id in enumerate(q_ref_ids):
        if ref_id in pool_id_to_idx:
            ref_idx = pool_id_to_idx[ref_id]
            global_scores[i, ref_idx] = float('-inf')
        else:
            raise KeyError(f"Query reference ID '{ref_id}' not found in pool IDs!")

    global_topk_indices = global_scores.topk(50, dim=-1).indices.cpu().tolist()

    # Convert retrieved indices to image IDs
    global_retrieved_ids = []
    for i in range(num_queries):
        ret_ids = [pool_ids[idx] for idx in global_topk_indices[i]]
        global_retrieved_ids.append(ret_ids)

    # Handle official TEST1 submission output
    if split == "test1":
        out_dir = Path(save_dir) if save_dir else Path.cwd() / "submissions"
        out_dir.mkdir(parents=True, exist_ok=True)

        recall_submission = {"version": "rc2", "metric": "recall"}
        subset_submission = {"version": "rc2", "metric": "recall_subset"}

        # Process each test query for both submissions
        for i in range(num_queries):
            pairid_str = str(q_pairids[i])
            ref_id = q_ref_ids[i]

            # 1. Global top 50
            top50 = global_retrieved_ids[i][:50]
            recall_submission[pairid_str] = top50

            # 2. Subset top 3 (strictly rank members minus reference)
            subset_members = [m for m in q_subset_ids[i] if m != ref_id]
            valid_subset_members = [m for m in subset_members if m in pool_id_to_idx]
            subset_indices = [pool_id_to_idx[m] for m in valid_subset_members]
            
            if len(subset_indices) == 0:
                raise ValueError(f"No valid subset candidates for query pairid {pairid_str}")

            sub_embs = pool_embs[subset_indices].to(device=device, dtype=torch.float32)
            query_e = q_embs[i:i+1].to(device=device, dtype=torch.float32)
            sub_scores = (query_e @ sub_embs.T).squeeze(0)

            top_sub_idx = sub_scores.topk(min(3, len(sub_scores))).indices.cpu().tolist()
            top3_sub_ids = [valid_subset_members[idx] for idx in top_sub_idx]
            subset_submission[pairid_str] = top3_sub_ids

        # Write files
        recall_path = out_dir / f"rec_{exp_name}.json"
        subset_path = out_dir / f"rec_subset_{exp_name}.json"

        with open(recall_path, "w", encoding="utf-8") as f:
            json.dump(recall_submission, f, indent=2)
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(subset_submission, f, indent=2)

        print(f"[CIRR TEST1] Saved global submission to: {recall_path}")
        print(f"[CIRR TEST1] Saved subset submission to: {subset_path}")

        # Post-serialization verification
        _validate_submission_json(recall_path, expected_pairids=q_pairids, is_subset=False, pool_ids_set=set(pool_ids))
        _validate_submission_json(subset_path, expected_pairids=q_pairids, is_subset=True, pool_ids_set=set(pool_ids))
        print("[CIRR TEST1] Submission files successfully validated!")

        return {}

    # Validation split metrics calculation
    r1, r5, r10, r25, r50 = 0, 0, 0, 0, 0
    sub_r1, sub_r2, sub_r3 = 0, 0, 0

    for i in range(num_queries):
        target_id = q_target_ids[i]
        ref_id = q_ref_ids[i]

        # Global hit checks
        ret_ids = global_retrieved_ids[i]
        if target_id in ret_ids[:1]:  r1 += 1
        if target_id in ret_ids[:5]:  r5 += 1
        if target_id in ret_ids[:10]: r10 += 1
        if target_id in ret_ids[:25]: r25 += 1
        if target_id in ret_ids[:50]: r50 += 1

        # Subset Recall checks
        subset_members = [m for m in q_subset_ids[i] if m != ref_id]
        if target_id not in subset_members:
            raise ValueError(f"Target ID '{target_id}' not found in query subset members (pairid={q_pairids[i]})!")

        valid_subset_members = [m for m in subset_members if m in pool_id_to_idx]
        subset_indices = [pool_id_to_idx[m] for m in valid_subset_members]
        sub_embs = pool_embs[subset_indices].to(device=device, dtype=torch.float32)
        query_e = q_embs[i:i+1].to(device=device, dtype=torch.float32)
        sub_scores = (query_e @ sub_embs.T).squeeze(0)

        sub_ranked_indices = sub_scores.sort(descending=True).indices.cpu().tolist()
        sub_ranked_ids = [valid_subset_members[idx] for idx in sub_ranked_indices]

        target_sub_rank = sub_ranked_ids.index(target_id) + 1
        if target_sub_rank <= 1: sub_r1 += 1
        if target_sub_rank <= 2: sub_r2 += 1
        if target_sub_rank <= 3: sub_r3 += 1

    metrics = {
        "R@1":        round((r1 / num_queries) * 100.0, 2),
        "R@5":        round((r5 / num_queries) * 100.0, 2),
        "R@10":       round((r10 / num_queries) * 100.0, 2),
        "R@25":       round((r25 / num_queries) * 100.0, 2),
        "R@50":       round((r50 / num_queries) * 100.0, 2),
        "RSubset@1":  round((sub_r1 / num_queries) * 100.0, 2),
        "RSubset@2":  round((sub_r2 / num_queries) * 100.0, 2),
        "RSubset@3":  round((sub_r3 / num_queries) * 100.0, 2),
        "CIRR_mean":  round(((r5 / num_queries + sub_r1 / num_queries) / 2.0) * 100.0, 2),
    }

    return metrics

def _validate_submission_json(json_path: Path, expected_pairids: List[Any], is_subset: bool, pool_ids_set: set):
    """Rigorous post-write verification of test submission files."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == "rc2", f"Submission missing 'version': 'rc2' in {json_path}"
    expected_metric = "recall_subset" if is_subset else "recall"
    assert data.get("metric") == expected_metric, f"Expected metric '{expected_metric}' in {json_path}"

    expected_pair_strs = set(str(pid) for pid in expected_pairids)
    keys_in_json = set(data.keys()) - {"version", "metric"}

    assert keys_in_json == expected_pair_strs, (
        f"Mismatch in pairids! Missing: {len(expected_pair_strs - keys_in_json)}, "
        f"Unexpected: {len(keys_in_json - expected_pair_strs)}"
    )

    expected_len = 3 if is_subset else 50
    for pid_str in keys_in_json:
        preds = data[pid_str]
        assert isinstance(preds, list), f"Predictions for {pid_str} must be a list"
        assert len(preds) == expected_len, f"Expected {expected_len} predictions for {pid_str}, got {len(preds)}"
        assert len(set(preds)) == expected_len, f"Duplicate candidate predictions found for {pid_str}"
        for cand_id in preds:
            assert isinstance(cand_id, str), f"Candidate ID {cand_id} must be a string, not path"
            assert "/" not in cand_id and "\\" not in cand_id, f"Candidate ID {cand_id} looks like a path!"
            assert cand_id in pool_ids_set, f"Candidate ID {cand_id} not present in test1 gallery!"
