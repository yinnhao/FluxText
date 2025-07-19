import numpy as np
import cv2
import argparse
import sys
import os
from t3_dataset import draw_glyph2
from PIL import Image, ImageFont

def render_glyph_image(
    mask: np.ndarray,
    text: str,
    width: int,
    height: int,
    font_path: str,
    font_size: int = 60,
    save_path: str = None
) -> np.ndarray:
    """
    从掩码中提取轮廓，并在指定尺寸内绘制glyph图像。
    可选地将图像保存到磁盘。

    参数：
        mask: numpy数组，二值图像，表示绘制区域
        text: 要绘制的文字内容
        width: 目标图像宽度
        height: 目标图像高度
        font_path: 字体路径
        font_size: 字体大小（默认60）
        save_path: 可选，若指定路径则保存图像为PNG

    返回：
        glyph_img: np.ndarray，绘制好的图像（float 类型，范围 0~1）
    """
    if mask.ndim != 2:
        raise ValueError("mask 应该是单通道二维图像")
    
    # 提取轮廓
    mask = mask.astype('uint8')
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到有效轮廓")

    # 强制转换轮廓数据类型为 int32（OpenCV 要求）
    contour = contours[0].astype(np.float32)
    
    # 加载字体
    selffont = ImageFont.truetype(font_path, size=font_size)

    # 绘制 glyph 图像（float 类型，值在 0~1）
    glyph_img = draw_glyph2(selffont, text, contour, scale=1, width=width, height=height)

    # 保存图像（可选）
    if save_path is not None:
        glyph_img_uint8 = (1.0 - glyph_img) * 255  # 转换为白底黑字
        glyph_img_uint8 = glyph_img_uint8.astype(np.uint8)[:, :, 0]
        glyph_pil = Image.fromarray(glyph_img_uint8, mode="L")
        glyph_pil.save(save_path)
        print(f"[✓] Glyph image saved to: {save_path}")
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # 从单通道转为三通道
        # mask_save_path = save_path.replace(".png", "_mask_rgb.png")
        # cv2.imwrite(mask_save_path, mask_rgb)

    return glyph_img


def load_text_mapping(text_file_path):
    """
    从文本文件中加载文件名和对应文本的映射关系
    
    参数：
        text_file_path: 文本文件路径，每行格式为 "文件名 文本内容"
    
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
        sys.exit(1)


def get_image_files(mask_dir):
    """
    获取目录中所有图像文件
    
    参数：
        mask_dir: 图像目录路径
    
    返回：
        list: 图像文件路径列表
    """
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in os.listdir(mask_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_extensions:
            image_files.append(filename)
    
    return sorted(image_files)


def process_single_image(mask_path, text, font_path, font_size, output_path):
    """
    处理单张图像
    
    参数：
        mask_path: 掩码图像路径
        text: 文本内容
        font_path: 字体路径
        font_size: 字体大小
        output_path: 输出路径
    
    返回：
        bool: 是否成功处理
    """
    try:
        # 读取掩码图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"错误：无法读取掩码图像: {mask_path}")
            return False
        
        # 从mask图像获取宽度和高度
        height, width = mask.shape
        
        # 生成glyph图像
        glyph = render_glyph_image(
            mask=mask,
            text=text,
            width=width,
            height=height,
            font_path=font_path,
            font_size=font_size,
            save_path=output_path
        )
        return True
        
    except Exception as e:
        print(f"错误：处理图像 {mask_path} 失败: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='从掩码图像生成文字glyph图像（支持批量处理）')
    parser.add_argument('--mask_path', type=str, required=True,
                       help='输入掩码图像路径（单个文件或文件夹）')
    parser.add_argument('--text', type=str, required=True,
                       help='文字内容（单个文本或txt文件路径）')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出路径（单个文件或文件夹）')
    parser.add_argument('--font_path', type=str, 
                       default='/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/font/Arial_Unicode.ttf',
                       help='字体文件路径')
    parser.add_argument('--font_size', type=int, default=100,
                       help='字体大小')
    
    args = parser.parse_args()
    
    # 检查字体文件是否存在
    if not os.path.exists(args.font_path):
        print(f"错误：字体文件不存在: {args.font_path}")
        sys.exit(1)
    
    # 判断是批量处理还是单个文件处理
    if os.path.isdir(args.mask_path):
        # 批量处理模式
        print("[INFO] 批量处理模式")
        
        # 检查文本文件
        if not os.path.isfile(args.text):
            print(f"错误：文本文件不存在: {args.text}")
            sys.exit(1)
        
        # 创建输出目录
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            print(f"[INFO] 创建输出目录: {args.output_path}")
        
        # 加载文本映射
        text_mapping = load_text_mapping(args.text)
        
        # 获取所有图像文件
        image_files = get_image_files(args.mask_path)
        if not image_files:
            print(f"错误：在目录 {args.mask_path} 中未找到图像文件")
            sys.exit(1)
        
        print(f"[INFO] 找到 {len(image_files)} 个图像文件")
        
        # 批量处理
        success_count = 0
        for image_file in image_files:
            # 获取不带扩展名的文件名
            base_name = os.path.splitext(image_file)[0]
            
            # 查找对应的文本
            if image_file not in text_mapping and base_name not in text_mapping:
                print(f"警告：未找到文件 {image_file} 对应的文本，跳过")
                continue
            
            text_content = text_mapping.get(image_file) or text_mapping.get(base_name)
            mask_file_path = os.path.join(args.mask_path, image_file)
            output_file_path = os.path.join(args.output_path, f"{base_name}_glyph.png")
            
            print(f"[INFO] 处理: {image_file} -> {text_content}")
            
            if process_single_image(mask_file_path, text_content, args.font_path, 
                                  args.font_size, output_file_path):
                success_count += 1
        
        print(f"[✓] 批量处理完成: 成功处理 {success_count}/{len(image_files)} 个文件")
    
    else:
        # 单个文件处理模式
        print("[INFO] 单个文件处理模式")
        
        # 检查输入文件是否存在
        if not os.path.exists(args.mask_path):
            print(f"错误：掩码图像文件不存在: {args.mask_path}")
            sys.exit(1)
        
        # 创建输出目录（如果输出路径包含目录）
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if process_single_image(args.mask_path, args.text, args.font_path, 
                              args.font_size, args.output_path):
            print(f"[✓] 成功生成glyph图像: {args.output_path}")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()