from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_qwen2vlm": ["Qwen2vlmConfig", "UrsaConfig"],
    "processing_qwen2vlm": ["Qwen2vlmProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_qwen2vlm"] = [
        "Qwen2vlmForConditionalGeneration",
        "Qwen2vlmPreTrainedModel",
        "UrsaForConditionalGeneration",
        "UrsaForTokenClassification"
    ]


if TYPE_CHECKING:
    from .configuration_qwen2vlm import Qwen2vlmConfig
    from .processing_qwen2vlm import Qwen2vlmProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qwen2vlm import (
            Qwen2vlmForConditionalGeneration,
            Qwen2vlmPreTrainedModel,
            UrsaForTokenClassification,
            UrsaForConditionalGeneration
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
