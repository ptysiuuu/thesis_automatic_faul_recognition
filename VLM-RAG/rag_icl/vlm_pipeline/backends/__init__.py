from .qwen import QwenVLBackend, PROCESSOR_FALLBACK, BASE_PROCESSOR
from .ollama import OllamaBackend
from .nvidia_mistral import NvidiaMistralBackend
from .gemini import GeminiBackend

PHI4_MODELS = {"microsoft/Phi-4-reasoning-vision-15B"}
INTERNVL_MODELS = {"OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B"}
QWEN25_MODELS = {"Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct"}
QWEN3VL_MODELS = {
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-Instruct",
    "Qwen/Qwen3-VL-235B-Instruct",
}
QWEN35_MODELS = {"Qwen/Qwen3.5-9B"}
GEMMA4_MODELS = {
    "google/gemma-4-12b-it",
    "google/gemma-4-31b-it",
}
NVIDIA_MODELS = {
    "mistralai/mistral-large-3-675b-instruct-2512",
}
GEMINI_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
}


def get_backend(
    model_name: str,
    enable_thinking: bool = False,
    use_video_mode: bool = False,
    quantize: str = "",
):
    lower_name = model_name.lower()
    if lower_name.startswith("nvidia:") or lower_name.startswith("nvidia/"):
        clean_name = model_name.split(":", 1)[-1]
        if clean_name.startswith("nvidia/"):
            clean_name = clean_name.split("/", 1)[-1]
        return NvidiaMistralBackend(model_name=clean_name)

    if (
        lower_name.startswith("gemini:")
        or lower_name.startswith("google:")
        or lower_name.startswith("google/")
    ):
        # Route google/gemma-4-* to Gemma4Backend, not Gemini
        if "gemma" in lower_name:
            from .gemma4 import Gemma4Backend

            return Gemma4Backend(
                model_name=model_name,
                enable_thinking=enable_thinking,
                quantize=quantize,
            )
        clean_name = model_name.split(":", 1)[-1]
        if clean_name.startswith("google/"):
            clean_name = clean_name.split("/", 1)[-1]
        return GeminiBackend(model_name=clean_name, use_video_mode=use_video_mode)

    if lower_name in {"mistral_nvidia", "nvidia_mistral"}:
        return NvidiaMistralBackend()

    if model_name in NVIDIA_MODELS:
        return NvidiaMistralBackend(model_name=model_name)

    if model_name in GEMINI_MODELS:
        return GeminiBackend(model_name=model_name, use_video_mode=use_video_mode)

    if model_name in GEMMA4_MODELS:
        from .gemma4 import Gemma4Backend

        return Gemma4Backend(
            model_name=model_name,
            enable_thinking=enable_thinking,
            quantize=quantize,
        )

    if "-Video" in model_name:
        from .qwen import QwenVLBackend as QwenVideoBackend

        clean_name = model_name.replace("-Video", "")
        return QwenVideoBackend(model_name=clean_name, use_video_mode=True)

    if model_name in PHI4_MODELS:
        from .phi4 import Phi4VisionBackend

        return Phi4VisionBackend(model_name=model_name)

    if model_name in INTERNVL_MODELS:
        from .internvl import InternVLBackend

        return InternVLBackend(model_name=model_name)

    if model_name in QWEN25_MODELS:
        quantize_4bit = model_name == "Qwen/Qwen2.5-VL-32B-Instruct"
        return QwenVLBackend(
            model_name=model_name,
            quantize_4bit=quantize_4bit,
            use_video_mode=use_video_mode,
        )

    if model_name in QWEN3VL_MODELS:
        from .qwen3vl import Qwen3VLBackend

        return Qwen3VLBackend(
            model_name=model_name,
            enable_thinking=enable_thinking,
            quantize=quantize,
        )

    if model_name in QWEN35_MODELS:
        from .qwen35 import Qwen35Backend

        return Qwen35Backend(model_name=model_name, enable_thinking=enable_thinking)

    return QwenVLBackend(model_name=model_name, use_video_mode=use_video_mode)
