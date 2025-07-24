# embed_utils.py
import torch
from transformers import AutoProcessor, Siglip2VisionModel, Siglip2TextModel
from constants import MODEL_ID, DEVICE
import warnings
import sys # Import sys to check Python version for flash_attn hint

_loaded_models = {}

def get_siglip_models_and_processor(device=DEVICE):
    """Loads and caches the SigLIP-2 processor, vision model, and text model."""
    if "processor" in _loaded_models:
        return (_loaded_models["processor"],
                _loaded_models["vision_model"],
                _loaded_models["text_model"])

    print(f"Loading SigLIP-2 ({MODEL_ID}) on {device}…")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model_kwargs = {"torch_dtype": torch.float16}
    if device.startswith("cuda"):
        try:
            # Attempt to import flash_attn, but it's optional
            # Flash-attn typically requires Python 3.8+
            if sys.version_info >= (3, 8):
                 import flash_attn  # noqa F401 - tells linters to ignore unused import warning
                 model_kwargs["attn_implementation"] = "flash_attention_2"
                 print("→ Attempting to use flash_attention_2 (if installed and compatible)")
            else:
                 warnings.warn("flash-attn requires Python 3.8+; skipping flash-attn.")
                 print("→ Python version < 3.8; skipping flash-attn.")

        except ImportError:
            warnings.warn("flash-attn not found. Using default attention mechanism.")
            print("→ flash-attn not found; using default attention")
        except Exception as e:
             warnings.warn(f"Error enabling flash_attention_2: {e}. Using default attention mechanism.")
             print(f"→ Error with flash_attention_2: {e}; using default attention")


    # Ensure models are loaded in evaluation mode and prevent gradients computation
    with torch.no_grad():
        vision_model = Siglip2VisionModel.from_pretrained(MODEL_ID, **model_kwargs).to(device).eval()
        text_model   = Siglip2TextModel.from_pretrained(MODEL_ID, **model_kwargs).to(device).eval()

    _loaded_models.update({
        "processor": processor,
        "vision_model": vision_model,
        "text_model": text_model,
    })
    return processor, vision_model, text_model

# Removed get_image_embedding and get_text_embedding as they are not used