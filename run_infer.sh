# python infer.py \
#     --model_path weights/model_multisize/pytorch_lora_weights.safetensors \
#     --config_path weights/model_multisize/config.yaml \
#     --image_path ./assets/hint_imgs.jpg \
#     --mask_path ./assets/hint.png \
#     --text "Hello World" \
#     --output_path output.jpg \
#     --save_glyph \
#     --glyph_output_path glyph_images/my_glyph.png


# ./generate_text_image.sh \
#     --model_path model.safetensors \
#     --config_path config.yaml \
#     --image_path input.jpg \
#     --mask_path mask.jpg \
#     --text "Hello" \
#     --output_path output.jpg

# # 带自定义prompt
# ./generate_text_image.sh \
#     --model_path model.safetensors \
#     --config_path config.yaml \
#     --image_path input.jpg \
#     --mask_path mask.jpg \
#     --text "Hello" \
#     --output_path output.jpg \
#     --prompt "A beautiful sign that reads 'Hello'"

# 保存glyph
./generate_text_image.sh \
    --model_path weights/model_multisize/pytorch_lora_weights.safetensors \
    --config_path weights/model_multisize/config.yaml \
    --image_path assets/hint_imgs.jpg \
    --mask_path assets/hint.png \
    --text "Hello" \
    --output_path output.jpg \
    --save_glyph \
    --glyph_output_path glyph_images/my_glyph.png