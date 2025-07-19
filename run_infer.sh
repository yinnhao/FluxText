python infer.py \
    --model_path weights/model_multisize/pytorch_lora_weights.safetensors \
    --config_path weights/model_multisize/config.yaml \
    --image_path ./assets/hint_imgs.jpg \
    --mask_path ./assets/hint.png \
    --text "Hello World" \
    --output_path output.jpg \
    --save_glyph \
    --glyph_output_path glyph_images/my_glyph.png