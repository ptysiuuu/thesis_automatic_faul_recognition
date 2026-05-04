from .qwen import QwenVLBackend\nfrom .internvl import InternVLBackend\nfrom .qwen35 import Qwen35Backend

def get_backend(model_name: str):
    # This acts as a factory. You can add your Phi-4 backend here later!
        if 'InternVL' in model_name:
        return InternVLBackend(model_name=model_name)
    if '3.5' in model_name:\n        return Qwen35Backend(model_name=model_name)\n    return QwenVLBackend(model_name=model_name)
