#!/bin/bash

echo "Starting CoCa Evaluations..."
python eval.py --config configs/coca_stage1.yaml --checkpoint outputs_2/PTbb_coca_large_stage1_llava_b256/best_checkpoint.pt
python eval.py --config configs/coca_stage2_after_stage1.yaml --checkpoint outputs_2/PTbb_coca_large_tuned_stage2_mtcir_b256/best_checkpoint.pt
python eval.py --config configs/coca_stage2_only_frozen.yaml --checkpoint path/to/coca/stage2_frozen/best_checkpoint.pt
python eval.py --config configs/coca_stage2_only_unfrozen.yaml --checkpoint outputs_2/PTbb_coca_large_tuned_only_stage2_mtcir_b248/best_checkpoint.pt

echo "Starting BLIP Evaluations..."
python eval.py --config configs/blip_stage1.yaml --checkpoint outputs/PTbb_blip_large_stage1_llava_b256/last_checkpoint.pt
python eval.py --config configs/blip_stage2_after_stage1.yaml --checkpoint outputs_2/PTbb_blip_large_stage2_mtcir_b256/
python eval.py --config configs/blip_stage2_only_frozen.yaml --checkpoint path/to/blip/stage2_frozen/best_checkpoint.pt
python eval.py --config configs/blip_stage2_only_unfrozen.yaml --checkpoint outputs_2/PTbb_blip_large_tuned_only_stage2_mtcir_b248/last_checkpoint.pt

echo "Starting CLIP Evaluations..."
python eval.py --config configs/clip.yaml --checkpoint outputs_2/PTbb_clip_large_stage1_llava_b256/last_checkpoint.pt
python eval.py --config configs/clip_stage2_after_stage1.yaml --checkpoint outputs_2/PTbb_clip_large_stage2_mtcir_b256/last_checkpoint.pt
python eval.py --config configs/clip_stage2_only_frozen.yaml --checkpoint path/to/clip/stage2_frozen/best_checkpoint.pt
python eval.py --config configs/clip_stage2_only_unfrozen.yaml --checkpoint outputs_2/PTbb_clip_large_tuned_only_stage2_mtcir_b256/last_checkpoint.pt

echo "All evaluations completed!"


python eval.py --config configs/coca_stage2_tuned_after_stage1.yaml --checkpoint outputs_2/PTbb_coca_large_tuned_stage2_mtcir_b256/best_checkpoint.pt
python eval.py --config configs/blip_stage2_after_stage1.yaml --checkpoint outputs_2/PTbb_blip_large_stage2_mtcir_b256/best_checkpoint.pt
python eval.py --config configs/blip_stage2_only_unfrozen.yaml --checkpoint outputs_2/PTbb_blip_large_tuned_only_stage2_mtcir_b248/best_checkpoint.pt
python eval.py --config configs/blip_stage1.yaml --checkpoint outputs/PTbb_blip_large_stage1_llava_b256/best_checkpoint.pt
python eval.py --config configs/coca_stage2_only_unfrozen.yaml --checkpoint outputs_2/PTbb_coca_large_tuned_only_stage2_mtcir_b248/best_checkpoint.pt


python eval.py --config configs/blip_stage2_tuned_after_stage1.yaml --checkpoint outputs_shy/PTbb_blip_large_tuned_stage2_mtcir_b248/best_checkpoint.pt
python eval.py --config configs/clip_stage2_tuned_after_stage1.yaml --checkpoint outputs_shy/PTbb_clip_large_tuned_stage2_mtcir_b248/best_checkpoint.pt
python eval.py --config configs/coca_stage2_after_stage1.yaml --checkpoint outputs_shy/PTbb_coca_large_stage2_mtcir_b256/best_checkpoint.pt
python eval.py --config configs/clip_stage2_after_stage1.yaml --checkpoint outputs_shy/PTbb_clip_large_stage2_mtcir_b256/best_checkpoint.pt
python eval.py --config configs/clip_stage1.yaml --checkpoint outputs_shy/PTbb_clip_large_stage1_llava_b256/best_checkpoint.pt
python eval.py --config configs/coca_stage1.yaml --checkpoint outputs_shy/PTbb_coca_large_stage1_llava_b256/best_checkpoint.pt
python eval.py --config configs/clip_stage2_only_unfrozen.yaml --checkpoint outputs_shy/PTbb_clip_large_tuned_only_stage2_mtcir_b256/best_checkpoint.pt




