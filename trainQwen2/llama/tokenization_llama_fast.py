# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from shutil import copyfile
from typing import Optional, Tuple

from tokenizers import processors

from tokenization_utils_fast import PreTrainedTokenizerFast
# from tokenization_llama_fast import PreTrainedTokenizerFast
# from tokenization_llama_fast import PreTrainedTokenizerFast
from transformers.utils import is_sentencepiece_available, logging
from transformers.utils.versions import require_version


require_version("tokenizers>=0.13.3")

if is_sentencepiece_available():
    from .tokenization_llama import LlamaTokenizer
else:
    LlamaTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class LlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    ```python
    >>> from transformers import LlamaTokenizerFast

    >>> tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622
            and #25224 which includes fixes to properly handle tokens that appear after special tokens.
            Make sure to also set `from_slow` to `True`.
            A simple example:

            - `legacy=True`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
            >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
            [1, 15043, 29871, 1, 869]
            ```
            - `legacy=False`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
            >>> tokenizer.encode("Hello <s>.")  # 29889 is '.'
            [1, 15043, 29871, 1, 29889]
            ```
            Checkout the [pull request](https://github.com/huggingface/transformers/pull/24565) for more details.
        add_prefix_space (`bool`, *optional*):
            Whether or not the tokenizer should automatically add a prefix space
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = LlamaTokenizer
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token=True,
        add_eos_token=False,
        use_default_system_prompt=False,
        legacy=None,
        add_prefix_space=None,
        **kwargs,
    ):
        print("开始处理token")
        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file"
                " you can ignore this message."
            )
            legacy = True
        self.legacy = legacy

        if add_prefix_space is not None:
            kwargs["from_slow"] = True

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_default_system_prompt=use_default_system_prompt,
            add_prefix_space=add_prefix_space,
            legacy=legacy,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.use_default_system_prompt = use_default_system_prompt
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        print("\n===== 开始更新后处理器 =====")
        
        # 处理 BOS Token
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        print(f"[DEBUG] BOS 配置检查 | add_bos_token={self.add_bos_token}, bos_token='{bos}', bos_token_id={bos_token_id}")
        
        if bos is None and self.add_bos_token:
            print("[ERROR] 配置冲突：启用了 add_bos_token 但未定义 bos_token！")
            raise ValueError("add_bos_token = True but bos_token = None")
            
        # 处理 EOS Token
        eos = self.eos_token
        eos_token_id = self.eos_token_id
        print(f"[DEBUG] EOS 配置检查 | add_eos_token={self.add_eos_token}, eos_token='{eos}', eos_token_id={eos_token_id}")
        
        if eos is None and self.add_eos_token:
            print("[ERROR] 配置冲突：启用了 add_eos_token 但未定义 eos_token！")
            raise ValueError("add_eos_token = True but eos_token = None")
            
        # 构建模板
        single_bos_part = f"{bos}:0 " if self.add_bos_token else ""
        single_eos_part = f" {eos}:0" if self.add_eos_token else ""
        single = f"{single_bos_part}$A:0{single_eos_part}"
        print(f"[DEBUG] 单序列模板生成 | 模板结构 = '{single}'")
        
        pair_bos_part = f" {bos}:1" if self.add_bos_token else ""
        pair_eos_part = f" {eos}:1" if self.add_eos_token else ""
        pair = f"{single}{pair_bos_part} $B:1{pair_eos_part}"
        print(f"[DEBUG] 双序列模板生成 | 模板结构 = '{pair}'")
        
        # 注册特殊Token
        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
            print(f"[DEBUG] 注册BOS特殊Token | token='{bos}', id={bos_token_id}")
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
            print(f"[DEBUG] 注册EOS特殊Token | token='{eos}', id={eos_token_id}")
            
        print(f"[DEBUG] 最终特殊Token列表: {special_tokens}")
        
        # 更新后处理器
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )
        print(f"[SUCCESS] 后处理器更新完成 | 单序列模板 = '{single}'")
        print(f"          | 双序列模板 = '{pair}'")
        print(f"          | 特殊Token数量 = {len(special_tokens)}")
        print("===== 后处理器更新结束 =====\n")

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    # TODO ArthurZ let's rely on the template processor instead, refactor all fast tokenizers
    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output


__all__ = ["LlamaTokenizerFast"]
