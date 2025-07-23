import numpy as np
import math
import os
from PIL import Image
import torch
import yaml
import argparse
from tqdm import tqdm

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

def load_text_mapping(text_file_path):
    """
    从文本文件中加载文件名和对应文本的映射关系
    
    参数：
        text_file_path: 文本文件路径，每行格式为 "文件名	文本内容"
    
    返回：
        dict: 文件名到文本内容的映射字典
    """
    text_mapping = {}
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 分割文件名和文本，使用第一个空格作为分隔符
                parts = line.split('	', 1)
                if len(parts) < 2:
                    print(f"警告：第{line_num}行格式不正确，跳过: {line}")
                    continue
                
                filename, text = parts
                text_mapping[filename] = text
        
        print(f"[INFO] 成功加载 {len(text_mapping)} 个文本映射")
        return text_mapping
    
    except Exception as e:
        print(f"错误：无法读取文本文件 {text_file_path}: {str(e)}")
        raise

def get_image_files(image_dir):
    """
    获取目录中所有图像文件
    
    参数：
        image_dir: 图像目录路径
    
    返回：
        list: 图像文件名列表
    """
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in os.listdir(image_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_extensions:
            image_files.append(filename)
    
    return sorted(image_files)

def process_single_image(model, pipe, config, hint_path, img_path, condition_path, 
                        prompt, output_path, device="cuda", seed=None):
    """
    处理单张图像
    """
    try:
        # Load and process images
        hint = Image.open(hint_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')
        condition_img = Image.open(condition_path).convert('RGB')

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
        condition_img = [condition_img, hint, img] # [h, w, 3]
        position_delta = [0, 0]
        
        condition = Condition(
            condition_type='word_fill',
            condition=condition_img,
            position_delta=position_delta,
        )
        
        # Setup generator with seed if provided
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        
        # Generate image
        res = generate_fill(
            pipe,
            prompt=[prompt, prompt],
            conditions=[condition],
            height=tgt_height,
            width=tgt_width,
            generator=generator,
            model_config=config.get("model", {}),
            default_lora=True
        )
        
        # Save output
        resized_img = res.images[0].resize((ori_width, ori_height))
        resized_img.save(output_path)
        return True
        
    except Exception as e:
        print(f"错误：处理图像时出错: {str(e)}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="FluxText Inference Script - 支持单张图片和批量处理")
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
        help="Text prompt for generation (单张图片模式) 或 文本映射文件路径 (批量模式)"
    )
    parser.add_argument(
        "--hint_path",
        type=str,
        required=True,
        help="Path to the hint/mask image (单张图片) 或 hint图片文件夹 (批量模式)"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the input image (单张图片) 或 输入图片文件夹 (批量模式)"
    )
    parser.add_argument(
        "--condition_path",
        type=str,
        required=True,
        help="Path to the condition image (单张图片) 或 condition图片文件夹 (批量模式)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出路径: 单张图片的输出文件名 或 批量模式的输出文件夹"
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
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="启用批量处理模式"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    print("正在初始化模型...")
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
    print("正在加载LoRA权重...")
    state_dict = load_file(args.lora_path)
    state_dict_new = {
        x.replace('lora_A', 'lora_A.default')
         .replace('lora_B', 'lora_B.default')
         .replace('transformer.', ''): v 
        for x, v in state_dict.items()
    }
    model.transformer.load_state_dict(state_dict_new, strict=False)
    pipe = model.flux_pipe
    print("模型初始化完成！")

    # Check if batch mode or single image mode
    if args.batch_mode:
        print("=== 批量处理模式 ===")
        
        # Validate paths for batch mode
        if not os.path.isdir(args.hint_path):
            raise ValueError(f"批量模式下 hint_path 必须是文件夹: {args.hint_path}")
        if not os.path.isdir(args.img_path):
            raise ValueError(f"批量模式下 img_path 必须是文件夹: {args.img_path}")
        if not os.path.isdir(args.condition_path):
            raise ValueError(f"批量模式下 condition_path 必须是文件夹: {args.condition_path}")
        if not os.path.isfile(args.prompt):
            raise ValueError(f"批量模式下 prompt 必须是文本文件: {args.prompt}")
        
        # Create output directory
        os.makedirs(args.output_path, exist_ok=True)
        print(f"输出目录: {args.output_path}")
        
        # Load text mapping
        text_mapping = load_text_mapping(args.prompt)
        
        # Get image files from hint directory (as reference)
        hint_files = get_image_files(args.hint_path)
        print(f"找到 {len(hint_files)} 个hint图片文件")
        
        # Process each image
        success_count = 0
        for hint_file in tqdm(hint_files, desc="处理图片"):
            base_name = os.path.splitext(hint_file)[0]
            
            # Check if we have text for this image
            if hint_file not in text_mapping and base_name not in text_mapping:
                print(f"警告：未找到文件 {hint_file} 对应的文本，跳过")
                continue
            
            prompt_text = text_mapping.get(hint_file) or text_mapping.get(base_name)
            
            # Build file paths
            hint_file_path = os.path.join(args.hint_path, hint_file)
            img_file_path = os.path.join(args.img_path, hint_file)
            condition_file_path = os.path.join(args.condition_path, hint_file.split(".")[0] + "_glyph.png")
            output_file_path = os.path.join(args.output_path, f"{base_name}_output.png")
            
            # Check if all required files exist
            if not os.path.exists(img_file_path):
                print(f"警告：输入图片文件不存在: {img_file_path}，跳过")
                continue
            if not os.path.exists(condition_file_path):
                print(f"警告：condition图片文件不存在: {condition_file_path}，跳过")
                continue
            
            # Process single image

            prompt_text = "An image with the following text in it: " + prompt_text
            if process_single_image(
                model, pipe, config, hint_file_path, img_file_path, 
                condition_file_path, prompt_text, output_file_path, 
                args.device, args.seed
            ):
                success_count += 1
            else:
                print(f"处理失败: {hint_file}")
        
        print(f"批量处理完成: 成功处理 {success_count}/{len(hint_files)} 个图片")
        
    else:
        print("=== 单张图片模式 ===")
        
        # Validate paths for single image mode
        if not os.path.isfile(args.hint_path):
            raise ValueError(f"单张图片模式下 hint_path 必须是文件: {args.hint_path}")
        if not os.path.isfile(args.img_path):
            raise ValueError(f"单张图片模式下 img_path 必须是文件: {args.img_path}")
        if not os.path.isfile(args.condition_path):
            raise ValueError(f"单张图片模式下 condition_path 必须是文件: {args.condition_path}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process single image
        if process_single_image(
            model, pipe, config, args.hint_path, args.img_path, 
            args.condition_path, args.prompt, args.output_path, 
            args.device, args.seed
        ):
            print(f"输出已保存到: {args.output_path}")
        else:
            print("处理失败")

if __name__ == "__main__":
    main()