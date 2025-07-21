#!/bin/bash

# 配置基础参数
CONFIG_PATH="weights/model_multisize/config.yaml"
LORA_PATH="weights/model_multisize/pytorch_lora_weights.safetensors"
PROMPT="text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/wenzi_2025-07-10_2025-07-16.txt"
HINT_PATH="text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/mask_vis"
IMG_PATH="text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/imgs"
CONDITION_PATH="text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/glyph"

# 基础输出目录
BASE_OUTPUT_DIR="text_edit/0710-0716-select-wise/wenzi_2025-07-10_2025-07-16/seed_test_results"

# 创建基础输出目录
mkdir -p "$BASE_OUTPUT_DIR"

echo "开始测试不同 seed 值的效果 (0-50)"
echo "结果将保存在: $BASE_OUTPUT_DIR"
echo "========================================"

# 记录开始时间
start_time=$(date +%s)

# 循环测试 seed 0 到 50
for seed in {18..50}; do
    echo "处理 seed: $seed"
    
    # 为每个 seed 创建单独的输出文件夹
    output_dir="$BASE_OUTPUT_DIR/seed_$seed"
    mkdir -p "$output_dir"
    
    # 运行推理
    python run_infer.py \
        --config_path "$CONFIG_PATH" \
        --lora_path "$LORA_PATH" \
        --prompt "$PROMPT" \
        --hint_path "$HINT_PATH" \
        --img_path "$IMG_PATH" \
        --condition_path "$CONDITION_PATH" \
        --output_path "$output_dir" \
        --batch_mode \
        --seed $seed
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✓ Seed $seed 处理完成，结果保存在: $output_dir"
    else
        echo "✗ Seed $seed 处理失败"
    fi
    
    echo "----------------------------------------"
done

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "========================================"
echo "所有 seed 测试完成！"
echo "总耗时: ${minutes}分${seconds}秒"
echo "结果保存在: $BASE_OUTPUT_DIR"
echo ""
echo "文件夹结构:"
echo "$BASE_OUTPUT_DIR/"
echo "├── seed_0/"
echo "├── seed_1/"
echo "├── seed_2/"
echo "├── ..."
echo "└── seed_50/"
echo ""
echo "每个文件夹包含对应 seed 值生成的所有图片结果" 