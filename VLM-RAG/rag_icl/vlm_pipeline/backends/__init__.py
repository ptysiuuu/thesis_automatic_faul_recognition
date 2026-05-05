from .qwen import QwenVLBackend, PROCESSOR_FALLBACK, BASE_PROCESSOR

PHI4_MODELS = {"microsoft/Phi-4-reasoning-vision-15B"}
INTERNVL_MODELS = {"OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B",
                   "OpenGVLab/InternVL2_5-8B", "OpenGVLab/InternVL2_5-14B"}
QWEN35_MODELS = {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-32B"}


def get_backend(model_name: str):
    """Auto-select correct backend based on model name."""
    if model_name in PHI4_MODELS:
        from .phi4 import Phi4VisionBackend
        return Phi4VisionBackend(model_name=model_name)
    if model_name in INTERNVL_MODELS:
        from .internvl import InternVLBackend
        return InternVLBackend(model_name=model_name)
    if model_name in QWEN35_MODELS:
        from .qwen35 import Qwen35Backend
        return Qwen35Backend(model_name=model_name)
    return QwenVLBackend(model_name=model_name)
