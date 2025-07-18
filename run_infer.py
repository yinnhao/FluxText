import numpy as np
from PIL import Image
import torch
import yaml

from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll
from safetensors.torch import load_file

config_path = "/root/paddlejob/workspace/env_run/zhuyinghao/TAIR/FluxText/weights/model_multisize/config.yaml"
lora_path = "/root/paddlejob/workspace/env_run/zhuyinghao/TAIR/FluxText/weights/model_multisize/pytorch_lora_weights.safetensors"
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

prompt = "lepto college of education, the written materials on the picture: LESOTHO , COLLEGE OF , RE BONA LESELI LESEL , EDUCATION ."
hint = Image.open("assets/hint.png").resize((512, 512)).convert('RGB')
img = Image.open("assets/hint_imgs.jpg").resize((512, 512))
condition_img = Image.open("assets/hint_imgs_word.png").resize((512, 512)).convert('RGB')
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
    height=512,
    width=512,
    generator=generator,
    model_config=config.get("model", {}),
    default_lora=True,
)
res.images[0].save('flux_fill.png')