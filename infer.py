import argparse
import math
import os
import sys
import yaml
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'eval')))
from eval.t3_dataset import draw_glyph2
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll

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

class FluxTextEditor:
    def __init__(self, model_path: str, config_path: str, font_path: str = './font/Arial_Unicode.ttf', skip_caption_model: bool = False):
        """
        Initialize the FluxText editor.
        
        Args:
            model_path: Path to the model checkpoint file
            config_path: Path to the config YAML file
            font_path: Path to the font file for glyph generation
            skip_caption_model: If True, skip loading the BLIP model for caption generation
        """
        self.font_path = font_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.skip_caption_model = skip_caption_model
        
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize font
        self.font = ImageFont.truetype(font_path, size=60)
        
        # Initialize generator
        self.generator = torch.Generator(device=self.device)
        
        # Initialize BLIP model for caption generation only if needed
        if not skip_caption_model:
            print("Loading BLIP model for caption generation...")
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blipmodel = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", 
                torch_dtype=torch.float16
            )
            self.blipmodel.to(self.device, torch.float16)
            print("BLIP model loaded successfully.")
        else:
            print("Skipping BLIP model loading (prompt provided).")
            self.processor = None
            self.blipmodel = None
        
        # Initialize the main model
        self.pipe, self.trainable_model = self._init_pipeline(model_path)
        
    def _init_pipeline(self, model_path: str):
        """Initialize the FluxText pipeline with trained weights."""
        training_config = self.config["train"]
        trainable_model = OminiModelFIll(
            flux_pipe_id=self.config["flux_path"],
            lora_config=training_config["lora_config"],
            device=self.device,
            dtype=getattr(torch, self.config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=self.config.get("model", {}),
            gradient_checkpointing=True,#training_config.get("gradient_checkpointing", False),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )

        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        state_dict1 = {
            x.replace('lora_A', 'lora_A.default')
            .replace('lora_B', 'lora_B.default')
            .replace('transformer.', ''): v 
            for x, v in state_dict.items()
        }
        trainable_model.transformer.load_state_dict(state_dict1, strict=False)

        pipe = trainable_model.flux_pipe
        return pipe, trainable_model
    
    def get_captions(self, image: np.ndarray, text: str) -> str:
        """Generate caption for the image with the specified text."""
        if self.skip_caption_model or self.processor is None or self.blipmodel is None:
            raise RuntimeError("Caption generation model not loaded. Cannot generate captions when skip_caption_model=True.")
        
        image_pil = Image.fromarray(image)
        inputs = self.processor(image_pil, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.blipmodel.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        caption = f'{generated_text}, that reads "{text}"'
        return caption
    
    def generate_glyph_from_mask(self, mask: np.ndarray, text: str, width: int, height: int) -> tuple:
        """
        Generate glyph image from mask and text.
        
        Args:
            mask: Binary mask array (0-255 values)
            text: Text to render
            width: Target width
            height: Target height
            
        Returns:
            tuple: (hint, glyphs) where hint is normalized mask and glyphs is the glyph image
        """
        
        mask = mask.astype('uint8')
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("未找到有效轮廓")

        # 强制转换轮廓数据类型为 int32（OpenCV 要求）
        contour = contours[0].astype(np.float32)
        hint = mask / 255
        glyph_scale = 1
        glyphs = draw_glyph2(self.font, text, contour, scale=glyph_scale, width=width, height=height)
        
        return hint, glyphs
    
    def edit_text_in_image(
        self, 
        original_image: Image.Image, 
        text: str, 
        mask_image: Image.Image,
        prompt: str = None,
        seed: int = 42,
        save_glyph: bool = False,
        glyph_output_path: str = None
    ) -> Image.Image:
        """
        Edit text in an image using the provided mask.
        
        Args:
            original_image: PIL Image - the original image to edit
            text: str - the text to insert/replace
            mask_image: PIL Image - binary mask indicating where to place text (white=text area, black=background)
            prompt: str - optional prompt override. If None, will auto-generate using BLIP
            seed: int - random seed for generation
            save_glyph: bool - whether to save the generated glyph image
            glyph_output_path: str - path to save the glyph image (required if save_glyph=True)
            
        Returns:
            PIL Image - the edited image with new text
        """
        # Convert inputs to proper format
        ori_width, ori_height = original_image.size
        
        # Calculate target dimensions
        num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
        aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
        close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
        tgt_height, tgt_width = aspect_ratio_dict[close_ratio]
        
        # Resize images to target dimensions
        img = original_image.resize((tgt_width, tgt_height))
        mask_img = mask_image.resize((tgt_width, tgt_height)).convert('RGB')
        
        # Convert mask to numpy for glyph generation
        mask_array = np.array(mask_img.convert('L'))  # Convert to grayscale
        
        # Generate glyph image
        hint, glyphs = self.generate_glyph_from_mask(mask_array, text, tgt_width, tgt_height)
        glyphs = glyphs[:,:,0]
        # Create glyph PIL image
        glyph_img = Image.fromarray(((1 - glyphs.astype('uint8')) * 255).astype('uint8'))
        if glyph_img.mode != 'RGB':
            glyph_img = glyph_img.convert('RGB')
        
        # Save glyph image if requested
        if save_glyph:
            if glyph_output_path is None:
                raise ValueError("glyph_output_path must be provided when save_glyph=True")
            
            # Create directory if it doesn't exist
            glyph_dir = os.path.dirname(glyph_output_path)
            if glyph_dir and not os.path.exists(glyph_dir):
                os.makedirs(glyph_dir)
            
            glyph_img.save(glyph_output_path)
            print(f"Glyph image saved to {glyph_output_path}")
        
        # Generate prompt if not provided
        if prompt is None:
            prompt = self.get_captions(np.array(img), text)
        
        return self._generate_image(prompt, img, glyph_img, mask_img, seed, tgt_height, tgt_width)
    
    def _generate_image(
        self, 
        prompt: str, 
        img: Image.Image, 
        glyph_img: Image.Image, 
        mask_img: Image.Image, 
        seed: int,
        tgt_height: int,
        tgt_width: int
    ) -> Image.Image:
        """Internal method to generate the final image."""
        # Prepare condition images
        hint = np.array(mask_img) / 255
        condition_img = np.array(glyph_img)
        condition_img = (255 - condition_img) / 255
        condition_img = [condition_img, hint, img]
        position_delta = [0, 0]
        
        # Create condition object
        condition = Condition(
            condition_type='word_fill',
            condition=condition_img,
            position_delta=position_delta,
        )
        
        # Set seed and generate
        self.generator.manual_seed(seed)
        res = generate_fill(
            self.pipe,
            prompt=prompt,
            conditions=[condition],
            height=tgt_height,
            width=tgt_width,
            generator=self.generator,
            model_config=self.config.get("model", {}),
            default_lora=True,
        )
        
        return res.images[0]

def main():
    """Example usage of the FluxTextEditor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--font_path", type=str, default="./font/Arial_Unicode.ttf", help="Path to font file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask image")
    parser.add_argument("--text", type=str, required=True, help="Text to insert")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output image")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # 新增参数：控制是否保存glyph image
    parser.add_argument("--save_glyph", action="store_true", help="Save the generated glyph image")
    parser.add_argument("--glyph_output_path", type=str, default=None, help="Path to save glyph image (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # 如果要保存glyph但没有指定路径，自动生成路径
    if args.save_glyph and args.glyph_output_path is None:
        # 基于输出图片路径生成glyph路径
        output_dir = os.path.dirname(args.output_path)
        output_name = os.path.splitext(os.path.basename(args.output_path))[0]
        args.glyph_output_path = os.path.join(output_dir, f"{output_name}_glyph.png")
    
    # Determine whether to skip caption model loading
    skip_caption_model = args.prompt is not None
    
    # Initialize editor
    editor = FluxTextEditor(
        model_path=args.model_path,
        config_path=args.config_path,
        font_path=args.font_path,
        skip_caption_model=skip_caption_model
    )
    
    # Load images
    original_image = Image.open(args.image_path)
    mask_image = Image.open(args.mask_path)
    
    # Edit text
    result_image = editor.edit_text_in_image(
        original_image=original_image,
        text=args.text,
        mask_image=mask_image,
        prompt=args.prompt,
        seed=args.seed,
        save_glyph=args.save_glyph,
        glyph_output_path=args.glyph_output_path
    )
    
    # Save result
    result_image.save(args.output_path)
    print(f"Result saved to {args.output_path}")

if __name__ == "__main__":
    main()