python run_infer.py \
    --config_path "weights/model_multisize/config.yaml" \
    --lora_path "weights/model_multisize/pytorch_lora_weights.safetensors" \
    --prompt "text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/wenzi_2025-07-10_2025-07-16.txt" \
    --hint_path "text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/mask_vis" \
    --img_path "text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/imgs" \
    --condition_path "text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/glyph" \
    --output_path "text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/flux_text" \
    --batch_mode
    --seed 0