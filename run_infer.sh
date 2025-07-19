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
img=./text_edit/0710-0716-select/wenzi_2025-07-10_2025-07-16/mask_vis/002_2025-07-15.jpeg
mask=./text_edit/0710-0716-select/wenzi_2025-07-10_2025-07-16/imgs/002_2025-07-15.jpeg

./generate_text_image.sh \
    --model_path weights/model_multisize/pytorch_lora_weights.safetensors \
    --config_path weights/model_multisize/config.yaml \
    --image_path $img \
    --mask_path $mask \
    --text Hello \
    --output_path output2.jpg \
    --save_glyph \
    --glyph_output_path glyph_images/my_glyph2.png