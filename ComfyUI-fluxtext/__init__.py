from .nodes.comfy_nodes import FLUXTextLoad, FLUXTextGenerate, FLUXTextLORALoad, FLUXTextGenerateBasic, FLUXFillTransformerLoader, FLUXTextMaskImage, FLUXTextAutoSize

# 注册节点
NODE_CLASS_MAPPINGS = {
    "FLUXTextLoad": FLUXTextLoad,
    "FLUXTextGenerate": FLUXTextGenerate,
    "FLUXTextLORALoad": FLUXTextLORALoad,
    "FLUXTextGenerateBasic": FLUXTextGenerateBasic,
    "FLUXFillTransformerLoader": FLUXFillTransformerLoader,
    "FLUXTextMaskImage": FLUXTextMaskImage,
    "FLUXTextAutoSize": FLUXTextAutoSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FLUXTextLoad": "Load FLUX-Text Model",
    "FLUXTextGenerate": "FLUX-Text Generate",
    "FLUXTextLORALoad": "Load FLUX-Text LoRA",
    "FLUXTextGenerateBasic": "FLUX-Text Basic Generate",
    "FLUXFillTransformerLoader": "Load FLUX-Fill transformer",
    "FLUXTextMaskImage": "FLUX-Text MaskImage",
    "FLUXTextAutoSize": "FLUX-Text AutoSize",
} 


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']