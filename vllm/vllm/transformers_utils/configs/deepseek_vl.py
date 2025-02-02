#

# Permission is hereby granted,free of charge, to any person obtaining a copy of

# this software and associated documentation files (the "Software"), to deal in

# the Software without restriction, including without limitation the rights to

# use, copy, modify, merge, publish, distribute,sublicense,and/or sell copies of

# the Software,and to permit persons to whom the Software is furnished to do so,

# subject to the following conditions:

#

# The above copyright notice and this permission notice shall be included in all

# copies or substantial portions of the Software.

#

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

# IMPLIED,INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS

# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR

# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER

# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN

# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




from transformers import AutoConfig, LlamaConfig, PretrainedConfig


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: dict = {}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = kwargs.get("params", {})

class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: dict = {}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = kwargs.get("params", {})


class DeepSeekMultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    language_config: LlamaConfig
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)
        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)
        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)
        self.text_config = self.language_config


AutoConfig.register("multi_modality", DeepSeekMultiModalityConfig)