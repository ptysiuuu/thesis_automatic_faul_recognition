from .qwen import QwenVLBackend, PROCESSOR_FALLBACK, BASE_PROCESSOR
from .qwen_video import QwenVideoBackend

PHI4_MODELS     = {"microsoft/Phi-4-reasoning-vision-15B"}
INTERNVL_MODELS = {"OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B"}
QWEN3VL_MODELS  = {"Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-2B-Instruct",
                   "Qwen/Qwen3-VL-4B-Instruct"}
QWEN35_MODELS   = {"Qwen/Qwen3.5-9B"}

def get_backend(model_name: str, enable_thinking: bool = False):
    if '-Video' in model_name:
        clean_name = model_name.replace('-Video', '')
        return QwenVideoBackend(model_name=clean_name)
    
    if model_name in PHI4_MODELS:
        from .phi4 import Phi4VisionBackend
        return Phi4VisionBackend(model_name=model_name)
        
    if model_name in INTERNVL_MODELS:
        from .internvl import InternVLBackend
        return InternVLBackend(model_name=model_name)
        
    if model_name in QWEN3VL_MODELS:
        from .qwen3vl import Qwen3VLBackend
        return Qwen3VLBackend(model_name=model_name)
        
    if model_name in QWEN35_MODELS:
        from .qwen35 import Qwen35Backend
        return Qwen35Backend(model_name=model_name, enable_thinking=enable_thinking)
        
    return QwenVLBackend(model_name=model_name)
