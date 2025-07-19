python run_infer.py \
    --config_path "weights/model_multisize/config.yaml" \
    --lora_path "weights/model_multisize/pytorch_lora_weights.safetensors" \
    --prompt "prompts.txt" \
    --hint_path "hint_images/" \
    --img_path "input_images/" \
    --condition_path "condition_images/" \
    --output_path "output_images/" \
    --batch_mode
    --seed 0