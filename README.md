# Implementation of FLUX-Text

FLUX-Text: A Simple and Advanced Diffusion Transformer Baseline for Scene Text Editing

<a href='https://amap-ml.github.io/FLUX-text/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2505.03329'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href="https://huggingface.co/GD-ML/FLUX-Text/"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<!-- <a ><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a> -->

> *[Rui Lan](https://scholar.google.com/citations?user=zwVlWXwAAAAJ&hl=zh-CN), [Yancheng Bai](https://scholar.google.com/citations?hl=zh-CN&user=Ilx8WNkAAAAJ&view_op=list_works&sortby=pubdate), [Xu Duan](https://scholar.google.com/citations?hl=zh-CN&user=EEUiFbwAAAAJ), [Mingxing Li](https://scholar.google.com/citations?hl=zh-CN&user=-pfkprkAAAAJ), [Lei Sun](https://allylei.github.io), [Xiangxiang Chu](https://scholar.google.com/citations?hl=zh-CN&user=jn21pUsAAAAJ&view_op=list_works&sortby=pubdate)*
> <br>
> ALibaba Group

<img src='assets/flux-text.png'>

## 📖 Overview
* **Motivation:** Scene text editing is a challenging task that aims to modify or add text in images while maintaining the fidelity of newly generated text and visual coherence with the background. The main challenge of this task is that we need to edit multiple line texts with diverse language attributes (e.g., fonts, sizes, and styles), language types (e.g., English, Chinese), and visual scenarios (e.g., poster, advertising, gaming).
* **Contribution:** We propose FLUX-Text, a novel text editing framework for editing multi-line texts in complex visual scenes. By incorporating a lightweight Condition Injection LoRA module, Regional text perceptual loss, and two-stage training strategy, we significantly significant improvements on both Chinese and English benchmarks.
<img src='assets/method.png'>

## News

- **2025-07-03**: 🔥 We have released our [pre-trained checkpoints](https://huggingface.co/GD-ML/FLUX-Text/) on Hugging Face! You can now try out FLUX-Text with the official weights.

- **2025-06-26**: ⭐️ Inference and evaluate code are released. Once we have ensured that everything is functioning correctly, the new model will be merged into this repository.

## Todo List
1. - [x] Inference code 
2. - [x] Pre-trained weights 
3. - [ ] Gradio demo
4. - [ ] ComfyUI
5. - [ ] Training code

## 🛠️ Installation

We recommend using Python 3.10 and PyTorch with CUDA support. To set up the environment:

```bash
# Create a new conda environment
conda create -n flux_text python=3.10
conda activate flux_text

# Install other dependencies
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
pip install Pillow==9.5.0
```

## 🤗 Model Introduction

FLUX-Text is an open-source version of the scene text editing model. FLUX-Text can be used for editing posters, emotions, and more. The table below displays the list of text editing models we currently offer, along with their foundational information.

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Model Name</th>
    <th style="text-align: center;">Image Resolution</th>
    <th style="text-align: center;">Memory Usage</th>
    <th style="text-align: center;">English Sen.Acc</th>
    <th style="text-align: center;">Chinese Sen.Acc</th>
    <th style="text-align: center;">Download Link</th>
  </tr>
  <tr>
    <th style="text-align: center;">FLUX-Text-512</th>
    <th style="text-align: center;">512*512</th>
    <th style="text-align: center;">34G</th>
    <th style="text-align: center;">0.8419</th>
    <th style="text-align: center;">0.7132</th>
    <th style="text-align: center;"><a href="https://huggingface.co/GD-ML/FLUX-Text/tree/main/model_512">🤗 HuggingFace</a></th>
  </tr>
  <tr>
    <th style="text-align: center;">FLUX-Text</th>
    <th style="text-align: center;">Multi Resolution</th>
    <th style="text-align: center;">34G for (512*512)</th>
    <th style="text-align: center;">0.8228</th>
    <th style="text-align: center;">0.7161</th>
    <th style="text-align: center;"><a href="https://huggingface.co/GD-ML/FLUX-Text/tree/main/model_multisize">🤗 HuggingFace</a></th>
  </tr>
</table>

## 🔥 Quick Start

Here's a basic example of using FLUX-Text:

```python
import numpy as np
from PIL import Image
import torch
import yaml

from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll
from safetensors.torch import load_file

config_path = ""
lora_path = ""
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
```

## 💪🏻  Training

## 📊 Evaluation

For [Anytext-benchmark](https://modelscope.cn/datasets/iic/AnyText-benchmark/summary), please set the **config_path**, **model_path**, **json_path**, **output_dir** in the `eval/gen_imgs_anytext.sh` and generate the text editing results.

```bash
bash eval/gen_imgs_anytext.sh
```

For `Sen.ACC, NED, FID and LPIPS` evaluation, use the scripts in the `eval` folder.

```bash
bash eval/eval_ocr.sh
bash eval/eval_fid.sh
bash eval/eval_lpips.sh
```

## 📈 Results

<img src='assets/method_result.png'>

## 🌹 Acknowledgement

Our work is primarily based on [OminiControl](https://github.com/Yuanshi9815/OminiControl), [AnyText](https://github.com/tyxsspa/AnyText), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Phantom](https://github.com/Phantom-video/Phantom). We are sincerely grateful for their excellent works.

## 📚 Citation

If you find our paper and code helpful for your research, please consider starring our repository ⭐ and citing our work ✏️.
```bibtex
@misc{lan2025fluxtext,
    title={FLUX-Text: A Simple and Advanced Diffusion Transformer Baseline for Scene Text Editing},
    author={Rui Lan and Yancheng Bai and Xu Duan and Mingxing Li and Lei Sun and Xiangxiang Chu},
    year={2025},
    eprint={2505.03329},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```