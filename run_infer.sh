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
# img=./text_edit/0710-0716-select/wenzi_2025-07-10_2025-07-16/mask_vis/002_2025-07-15.jpeg
# mask=./text_edit/0710-0716-select/wenzi_2025-07-10_2025-07-16/imgs/002_2025-07-15.jpeg

# ./generate_text_image.sh \
#     --model_path weights/model_multisize/pytorch_lora_weights.safetensors \
#     --config_path weights/model_multisize/config.yaml \
#     --image_path $img \
#     --mask_path $mask \
#     --text Hello \
#     --output_path output2.jpg \
#     --save_glyph \
#     --glyph_output_path glyph_images/my_glyph2.png

python run_infer.py \
    --config_path "weights/model_multisize/config.yaml" \
    --lora_path "weights/model_multisize/pytorch_lora_weights.safetensors" \
    --prompt "An image with the following text in it: ID.3冲量底价" \
    --hint_path "/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/mask_vis/004_2025-07-12.jpeg" \
    --img_path "/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/imgs/004_2025-07-12.jpeg" \
    --condition_path "/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/glyph/004_2025-07-12_glyph.png" \
    --output_path "my_output_seed_14.png" \
    --seed 14


# python run_infer.py \
#     --config_path "weights/model_multisize/config.yaml" \
#     --lora_path "weights/model_multisize/pytorch_lora_weights.safetensors" \
#     --prompt "prompts.txt" \
#     --hint_path "hint_images/" \
#     --img_path "input_images/" \
#     --condition_path "condition_images/" \
#     --output_path "output_images/" \
#     --batch_mode