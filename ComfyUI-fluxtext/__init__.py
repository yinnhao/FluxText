from .nodes.comfy_nodes import FLUXTextLoad, FLUXTextGenerate

# 注册节点
NODE_CLASS_MAPPINGS = {
    "FLUXTextLoad": FLUXTextLoad,
    "FLUXTextGenerate": FLUXTextGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FLUXTextLoad": "Load FLUX-Text Model",
    "FLUXTextGenerate": "FLUX-Text Generate",
} 


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']