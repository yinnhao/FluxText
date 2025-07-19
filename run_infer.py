import numpy as np
import math
import os
from PIL import Image
import torch
import yaml

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

config_path = "/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/weights/model_multisize/config.yaml"
lora_path = "/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/weights/model_multisize/pytorch_lora_weights.safetensors"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
model = OminiModelFIll(
        flux_pipe_id=config["flux_path"],
        lora_config=config["train"]["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=True,
        byt5_encoder_config=None,
    )

state_dict = load_file(lora_path)
state_dict_new = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
model.transformer.load_state_dict(state_dict_new, strict=False)
pipe = model.flux_pipe

# prompt = "lepto college of education, the written materials on the picture: LESOTHO , COLLEGE OF , RE BONA LESELI LESEL , EDUCATION ."
# hint = Image.open("assets/hint.png").resize((512, 512)).convert('RGB')
# img = Image.open("assets/hint_imgs.jpg").resize((512, 512))
# condition_img = Image.open("assets/hint_imgs_word.png").resize((512, 512)).convert('RGB')

# height = 775
# width = 581

# hint_path = "/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/eval/glyph_test_mask_rgb.png"
# img_path = "text_edit/0710-0716-select/wenzi_2025-07-10_2025-07-16/imgs/002_2025-07-15.jpeg"
# condition_path = "/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/eval/glyph_test.png"


hint_path = "assets/hint2.jpg"
img_path = "assets/hint_imgs2.jpg"
condition_path = "assets/hint_imgs_word2.jpg"

prompt = "Car poster, that reads: ID.3冲量底价"
hint = Image.open(hint_path).convert('RGB')
img = Image.open(img_path).convert('RGB')
condition_img = Image.open(condition_path).convert('RGB')

ori_width, ori_height = img.size
num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
tgt_height, tgt_width = aspect_ratio_dict[close_ratio]

hint = hint.resize((tgt_width, tgt_height)).convert('RGB')
img = img.resize((tgt_width, tgt_height))
condition_img = condition_img.resize((tgt_width, tgt_height)).convert('RGB')


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
generator = torch.Generator(device="cuda")
res = generate_fill(
    pipe,
    prompt=prompt,
    conditions=[condition],
    height=tgt_height,
    width=tgt_width,
    generator=generator,
    model_config=config.get("model", {}),
    default_lora=True,
)
res.images[0].save('flux_fill2.png')