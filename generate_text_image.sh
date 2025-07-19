#!/bin/bash

# FluxText Pipeline: Caption Generation + Image Generation
set -e

# Default values
FONT_PATH="./font/Arial_Unicode.ttf"
SEED=42
TEMP_DIR="/tmp/fluxtext_$$"

usage() {
    cat << EOF
Usage: $0 --model_path MODEL --config_path CONFIG --image_path IMAGE --mask_path MASK --text TEXT --output_path OUTPUT [OPTIONS]

Required:
  --model_path PATH       Model checkpoint file
  --config_path PATH      Config YAML file  
  --image_path PATH       Input image
  --mask_path PATH        Mask image
  --text TEXT             Text to insert
  --output_path PATH      Output image path

Optional:
  --prompt TEXT           Custom prompt (skip caption generation)
  --font_path PATH        Font file (default: $FONT_PATH)
  --seed NUMBER           Random seed (default: $SEED)
  --save_glyph           Save glyph image
  --glyph_output_path PATH  Glyph save path
  -h, --help             Show help

Example:
  $0 --model_path model.safetensors --config_path config.yaml \\
     --image_path input.jpg --mask_path mask.jpg --text "Hello" \\
     --output_path output.jpg --save_glyph
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --config_path) CONFIG_PATH="$2"; shift 2 ;;
        --image_path) IMAGE_PATH="$2"; shift 2 ;;
        --mask_path) MASK_PATH="$2"; shift 2 ;;
        --text) TEXT="$2"; shift 2 ;;
        --output_path) OUTPUT_PATH="$2"; shift 2 ;;
        --prompt) CUSTOM_PROMPT="$2"; shift 2 ;;
        --font_path) FONT_PATH="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --save_glyph) SAVE_GLYPH=true; shift ;;
        --glyph_output_path) GLYPH_OUTPUT_PATH="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required arguments
for arg in MODEL_PATH CONFIG_PATH IMAGE_PATH MASK_PATH TEXT OUTPUT_PATH; do
    if [ -z "${!arg}" ]; then
        echo "Error: Missing required argument --${arg,,}"
        usage
    fi
done

# Check files exist
for file in "$MODEL_PATH" "$CONFIG_PATH" "$IMAGE_PATH" "$MASK_PATH"; do
    [ ! -f "$file" ] && { echo "Error: File not found: $file"; exit 1; }
done

# Setup
mkdir -p "$TEMP_DIR"
trap "rm -rf '$TEMP_DIR'" EXIT

echo "=== FluxText Pipeline ==="
echo "Input: $IMAGE_PATH | Text: '$TEXT' | Output: $OUTPUT_PATH"

# Step 1: Generate prompt (if not provided)
if [ -z "$CUSTOM_PROMPT" ]; then
    echo "Generating caption..."
    CAPTION_FILE="$TEMP_DIR/caption.txt"
    
    python caption_generator.py \
        --image_path "$IMAGE_PATH" \
        --text "$TEXT" \
        --output_file "$CAPTION_FILE"
    
    PROMPT=$(cat "$CAPTION_FILE")
    echo "Generated: $PROMPT"
else
    PROMPT="$CUSTOM_PROMPT"
    echo "Using custom prompt: $PROMPT"
fi

# Step 2: Generate image
echo "Generating image..."
ARGS=(
    --model_path "$MODEL_PATH"
    --config_path "$CONFIG_PATH" 
    --image_path "$IMAGE_PATH"
    --mask_path "$MASK_PATH"
    --text "$TEXT"
    --output_path "$OUTPUT_PATH"
    --prompt "$PROMPT"
    --font_path "$FONT_PATH"
    --seed "$SEED"
)

[ "$SAVE_GLYPH" = true ] && ARGS+=(--save_glyph)
[ -n "$GLYPH_OUTPUT_PATH" ] && ARGS+=(--glyph_output_path "$GLYPH_OUTPUT_PATH")

python infer.py "${ARGS[@]}"

echo "✓ Done! Output saved to: $OUTPUT_PATH" 