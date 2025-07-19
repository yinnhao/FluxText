import numpy as np
import math
import os
from PIL import Image
import torch
import yaml
import argparse

from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll
from safetensors.torch import load_file


ASPECT_RATIO_LD_LIST = [  # width:height
    "2.39:1",  # cinemascope, 2.39
    "2:1",  # rare, 2
    "16:9",  # rare, 1.89
    "1.85:1",  # american widescreen, 1.85
    "9:16",  # popular, 1.78
    "5:8",  # rare, 1.6
    "3:2",  # rare, 1.5
    "4:3",  # classic, 1.33
    "1:1",  # square
]

RESOLUTIONS = [512, 768, 1024]
PIXELS = [512 * 512, 768 * 768, 1024 * 1024]

def get_ratio(name: str) -> float:
    width, height = map(float, name.split(":"))
    return height / width

def get_closest_ratio(height: float, width: float, ratios: dict) -> str:
    aspect_ratio = height / width
    closest_ratio = min(
        ratios, key=lambda ratio: abs(aspect_ratio - get_ratio(ratio))
    )
    return closest_ratio

def get_aspect_ratios_dict(
    total_pixels: int = 256 * 256, training: bool = True
) -> dict[str, tuple[int, int]]:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    aspect_ratios_dict = {}
    aspect_ratios_vertical_dict = {}
    for ratio in ASPECT_RATIO_LD_LIST:
        width_ratio, height_ratio = map(float, ratio.split(":"))
        width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
        height = int((total_pixels / width) // D) * D

        if training:
            # adjust aspect ratio to match total pixels
            diff = abs(height * width - total_pixels)
            candidate = [
                (height - D, width),
                (height + D, width),
                (height, width - D),
                (height, width + D),
            ]
            for h, w in candidate:
                if abs(h * w - total_pixels) < diff:
                    height, width = h, w
                    diff = abs(h * w - total_pixels)

        # remove duplicated aspect ratio
        if (height, width) not in aspect_ratios_dict.values() or not training:
            aspect_ratios_dict[ratio] = (height, width)
            vertial_ratios = ":".join(ratio.split(":")[::-1])
            aspect_ratios_vertical_dict[vertial_ratios] = (width, height)

    aspect_ratios_dict.update(aspect_ratios_vertical_dict)

    return aspect_ratios_dict

def parse_args():
    parser = argparse.ArgumentParser(description="FluxText Inference Script")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config YAML file"
    )
    parser.add_argument(
        "--lora_path", 
        type=str,
        required=True,
        help="Path to the LoRA weights safetensors file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--hint_path",
        type=str,
        required=True,
        help="Path to the hint/mask image"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--condition_path",
        type=str,
        required=True,
        help="Path to the condition image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="flux_fill_output.png",
        help="Path to save the output image (default: flux_fill_output.png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = OminiModelFIll(
        flux_pipe_id=config["flux_path"],
        lora_config=config["train"]["lora_config"],
        device=args.device,
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=True,
        byt5_encoder_config=None,
    )

    # Load LoRA weights
    state_dict = load_file(args.lora_path)
    state_dict_new = {
        x.replace('lora_A', 'lora_A.default')
         .replace('lora_B', 'lora_B.default')
         .replace('transformer.', ''): v 
        for x, v in state_dict.items()
    }
    model.transformer.load_state_dict(state_dict_new, strict=False)
    pipe = model.flux_pipe

    # Load and process images
    hint = Image.open(args.hint_path).convert('RGB')
    img = Image.open(args.img_path).convert('RGB')
    condition_img = Image.open(args.condition_path).convert('RGB')

    # Calculate target dimensions
    ori_width, ori_height = img.size
    num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
    aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
    close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
    tgt_height, tgt_width = aspect_ratio_dict[close_ratio]

    # Resize images
    hint = hint.resize((tgt_width, tgt_height)).convert('RGB')
    img = img.resize((tgt_width, tgt_height))
    condition_img = condition_img.resize((tgt_width, tgt_height)).convert('RGB')

    # Process condition images
    hint = np.array(hint) / 255
    condition_img = np.array(condition_img)
    condition_img = (255 - condition_img) / 255
    condition_img = [condition_img, hint, img]
    position_delta = [0, 0]
    
    condition = Condition(
        condition_type='word_fill',
        condition=condition_img,
        position_delta=position_delta,
    )
    
    # Setup generator with seed if provided
    generator = torch.Generator(device=args.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    
    # Generate image
    res = generate_fill(
        pipe,
        prompt=args.prompt,
        conditions=[condition],
        height=tgt_height,
        width=tgt_width,
        generator=generator,
        model_config=config.get("model", {}),
        default_lora=True,
    )
    
    # Save output
    res.images[0].save(args.output_path)
    print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    main()