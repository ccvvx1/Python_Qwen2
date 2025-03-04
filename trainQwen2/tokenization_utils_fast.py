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
"""
Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
see tokenization_utils.py
"""

import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import tokenizers.pre_tokenizers as pre_tokenizers_fast
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.integrations.ggml import convert_gguf_tokenizer
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from tokenization_utils import PreTrainedTokenizer
from tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, add_end_docstrings, logging


logger = logging.get_logger(__name__)

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
TIKTOKEN_VOCAB_FILE = "tokenizer.model"

# Slow tokenizers have an additional added tokens files
ADDED_TOKENS_FILE = "added_tokens.json"

INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""

MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE, "vocab_file": TIKTOKEN_VOCAB_FILE}


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class: PreTrainedTokenizer = None

    def __init__(self, *args, **kwargs):
        print("\n===== 开始初始化Fast分词器 =====")
        print("[DEBUG] 输入参数概览:")
        print(f"  args长度: {len(args)}, kwargs关键字: {list(kwargs.keys())}")

        # 提取关键参数并打印
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        print(f"[PARAM] tokenizer_object: {'存在' if tokenizer_object else 'None'}")

        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        print(f"[PARAM] __slow_tokenizer: {'存在' if slow_tokenizer else 'None'}")

        gguf_file = kwargs.pop("gguf_file", None)
        print(f"[PARAM] gguf_file: {gguf_file or '未提供'}")

        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        print(f"[PARAM] tokenizer_file: {fast_tokenizer_file or '未提供'}")

        from_slow = kwargs.pop("from_slow", False)
        print(f"[PARAM] from_slow: {from_slow}")

        print("Q:为什么要加这些字段？字段从什么位置过来？")
        print("A:加一些特定词汇，字段经过token的from_pretrained函数读取tokenizer_config.json配置文件获取到")
        import sys, os
        print(f"\nQA跳转 File \"{os.path.abspath(__file__)}\", line {sys._getframe().f_lineno}")
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", {})
        print(f"[PARAM] added_tokens_decoder条目数: {len(added_tokens_decoder)}","具体内容：", added_tokens_decoder)

        # 处理前缀空格参数
        self.add_prefix_space = kwargs.get("add_prefix_space", False)
        print(f"[CONFIG] add_prefix_space: {self.add_prefix_space}")

        # 检查慢速分词器兼容性
        if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
            print("[ERROR] 无法从慢速分词器实例化：缺少sentencepiece依赖！")
            raise ValueError("Cannot instantiate...")

        # 初始化路径判断
        if tokenizer_object is not None:
            print("\n[BRANCH 1] 从现有分词器对象深度复制")
            print(f"  原始对象类型: {type(tokenizer_object).__name__}")
            fast_tokenizer = copy.deepcopy(tokenizer_object)
            print(f"  复制后对象ID: {id(fast_tokenizer)} (原始ID: {id(tokenizer_object)})")

        elif fast_tokenizer_file is not None and not from_slow:
            print(f"\n[BRANCH 2] 从文件加载: {fast_tokenizer_file}")
            try:
                print("  尝试加载tokenizers库序列化文件...")
                print("Q:从系统tokenizers函数已经可以获取到配置文件的add_specail_tokens字段，为什么还得在token的from_pretrained函数提前获取add_special_tokens内容？")
                print("A:因为这里的tokenizers是从tokenizer.json读取数据，而传进来的add_special_tokens数据来自tokenizr_config.json，来源不一样，最后得把两者合并")
                import sys, os
                print(f"\nQA跳转 File \"{os.path.abspath(__file__)}\", line {sys._getframe().f_lineno}")
                fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
                print(f"  加载成功！分词器类型: {type(fast_tokenizer).__name__}")
                print(f"  初始词汇量: {fast_tokenizer.get_vocab_size()}", "分解器内容：", fast_tokenizer)
            except Exception as e:
                print(f"[ERROR] 文件加载失败: {str(e)}")
                raise
        elif slow_tokenizer:
            print(f"\n[BRANCH 3] 转换慢速分词器: {type(slow_tokenizer).__name__}")
            print("  原始慢速分词器配置:")
            print(f"    vocab大小: {len(slow_tokenizer.vocab)}")
            print(f"    特殊Token: {slow_tokenizer.all_special_tokens}")
            try:
                fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
                print("  转换成功！生成的分词器:")
                print(f"    类型: {type(fast_tokenizer).__name__}")
                print(f"    新词汇量: {fast_tokenizer.get_vocab_size()}")
            except Exception as e:
                print(f"[ERROR] 转换失败: {str(e)}")
                raise

        elif gguf_file is not None:
            print(f"\n[BRANCH 4] 处理GGUF模型文件: {gguf_file}")
            try:
                print("  加载GGUF检查点...")
                gguf_param = load_gguf_checkpoint(kwargs.get("vocab_file"))
                print(f"  模型架构: {gguf_param['config']['model_type']}")
                print(f"  分词器配置键: {list(gguf_param['tokenizer'].keys())}")
                
                architecture = gguf_param["config"]["model_type"]
                tokenizer_dict = gguf_param["tokenizer"]
                tokenizer_config = gguf_param["tokenizer_config"]
                
                print("  开始转换GGUF分词器...")
                fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(architecture, tokenizer_dict)
                print(f"  获得额外参数: {list(additional_kwargs.keys())}")
                
                kwargs.update(tokenizer_config)
                if len(additional_kwargs) > 0:
                    kwargs.update(additional_kwargs)
                    print("  合并更新参数到kwargs")
            except Exception as e:
                print(f"[ERROR] GGUF处理失败: {str(e)}")
                raise

        elif self.slow_tokenizer_class is not None and slow_tokenizer is not False:
            print(f"\n[BRANCH 5] 动态创建慢速分词器: {self.slow_tokenizer_class.__name__}")
            print("  初始化参数:")
            print(f"    args: {args}")
            print(f"    kwargs: {kwargs}")
            
            try:
                slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
                print("  慢速分词器创建成功，开始转换...")
                fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
                print("  转换完成")
            except Exception as e:
                print(f"[ERROR] 慢速分词器初始化失败: {str(e)}")
                raise

        elif not slow_tokenizer:
            print("\n[BRANCH 6] 使用TikToken兼容模式")
            self.vocab_file = kwargs.get("vocab_file", None)
            print(f"  使用词汇表文件: {self.vocab_file}")
            
            self.additional_special_tokens = kwargs.get("additional_special_tokens", [])
            print(f"  额外特殊Token: {self.additional_special_tokens}")
            
            try:
                print("  尝试TikToken转换...")
                fast_tokenizer = convert_slow_tokenizer(self, from_tiktoken=True)
                print("  TikToken转换成功")
            except Exception as e:
                print(f"[ERROR] TikToken转换失败: {str(e)}")
                raise

        else:
            print("\n[ERROR] 无法匹配任何初始化路径！")
            print("  剩余参数:")
            print(f"    args: {args}")
            print(f"    kwargs: {kwargs}")
            raise ValueError("Couldn't instantiate the backend tokenizer...")

        print("[初始化] 开始设置Fast Tokenizer参数")
        self._tokenizer = fast_tokenizer
        print(f"[配置] 已绑定Fast Tokenizer对象: {type(self._tokenizer)}")

        # 慢速Tokenizer兼容逻辑
        if slow_tokenizer is not None:
            print("[兼容] 检测到Slow Tokenizer，合并初始化参数")
            kwargs.update(slow_tokenizer.init_kwargs)
            print(f"[参数] 更新后kwargs: {list(kwargs.keys())}")

        self._decode_use_source_tokenizer = False
        print(f"[解码] 设置解码不使用源分词器: {self._decode_use_source_tokenizer}")

        # 截断配置
        _truncation = self._tokenizer.truncation
        if _truncation is not None:
            print("\n[截断] 启用截断策略，参数详情:")
            print(f"  - 最大长度(max_length): {_truncation.get('max_length', '未设置')}")
            print(f"  - 方向(direction): {_truncation.get('direction', '未设置')}")
            print(f"  - 步长(stride): {_truncation.get('stride', '未设置')}")
            print(f"  - 策略(strategy): {_truncation.get('strategy', '未设置')}")
            
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault("max_length", _truncation["max_length"])
            kwargs.setdefault("truncation_side", _truncation["direction"])
            kwargs.setdefault("stride", _truncation["stride"])
            kwargs.setdefault("truncation_strategy", _truncation["strategy"])
            print("[截断] 参数已注入kwargs:", {k: kwargs[k] for k in ["max_length", "truncation_side", "stride", "truncation_strategy"]})
        else:
            print("[截断] 未检测到截断配置，禁用截断")
            self._tokenizer.no_truncation()

        # 填充配置
        _padding = self._tokenizer.padding
        if _padding is not None:
            print("\n[填充] 启用填充策略，参数详情:")
            print(f"  - 填充符(pad_token): {_padding.get('pad_token', '未设置')}")
            print(f"  - 方向(direction): {_padding.get('direction', '未设置')}")
            print(f"  - 最大长度(length): {_padding.get('length', '未设置')}")
            print(f"  - 倍数对齐(pad_to_multiple_of): {_padding.get('pad_to_multiple_of', '未设置')}")
            
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault("pad_token", _padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", _padding["pad_type_id"])
            kwargs.setdefault("padding_side", _padding["direction"])
            kwargs.setdefault("max_length", _padding["length"])
            kwargs.setdefault("pad_to_multiple_of", _padding["pad_to_multiple_of"])
            print("[填充] 参数已注入kwargs:", {k: kwargs[k] for k in ["pad_token", "padding_side", "max_length", "pad_to_multiple_of"]})
        else:
            print("[填充] 未检测到填充配置，禁用填充")

        # 调用父类初始化
        print("\n[继承] 执行父类初始化，最终参数:")
        for k, v in kwargs.items():
            print(f"  - {k}: {v}" if len(str(v)) < 50 else f"  - {k}: ...（长度{len(str(v))}）")
        super().__init__(**kwargs)

        # 特殊Token处理
        print("\n[特殊标记] 配置编码解码行为")
        self._tokenizer.encode_special_tokens = self.split_special_tokens
        # print(f"Encode特殊标记方法绑定: {self.split_special_tokens.__name__}")

        # 新增Token去重逻辑
        added_tokens_decoder_hash = {hash(repr(token)) for token in self.added_tokens_decoder}
        print(f"\n[去重] 现有已添加Token哈希值数量: {len(added_tokens_decoder_hash)}")
        
        tokens_to_add = [
            token
            for index, token in sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if hash(repr(token)) not in added_tokens_decoder_hash
        ]
        print(f"[新增] 需要添加的唯一Token数量: {len(tokens_to_add)}")

        # 合并特殊Token
        print("Q：新增的加密token和解密token是一样的？")
        print("A：是的，从代码上可以看到是一样的")
        import sys, os
        print(f"\nQA跳转 File \"{os.path.abspath(__file__)}\", line {sys._getframe().f_lineno}")
        encoder = list(self.added_tokens_encoder.keys()) + [str(token) for token in tokens_to_add]
        print(f"[合并] 当前编码器总Token数: {len(encoder)}")
        
        special_tokens = [
            token 
            for token in self.all_special_tokens_extended 
            if token not in encoder and token not in tokens_to_add
        ]
        tokens_to_add += special_tokens
        print(f"[特殊] 追加预定义特殊Token数量: {len(special_tokens)}")

        if len(tokens_to_add) > 0:
            print(f"\n[操作] 开始添加{len(tokens_to_add)}个Token到分词器")
            for i, token in enumerate(tokens_to_add, 1):
                is_special = (
                    (token.special or str(token) in self.all_special_tokens)
                    if isinstance(token, AddedToken)
                    else str(token) in self.all_special_tokens
                )
                print(f"  Token {i}: {str(token)[:20]}... | 是否特殊: {is_special}")
            self.add_tokens(tokens_to_add)
        else:
            print("[无操作] 没有需要添加的新Token")

        # 配置文件加载
        try:
            print("\n[配置] 尝试加载tokenizer.json文件")
            tokenizer_config = json.load(open("tokenizer.json"))
            print(f"  - 文件版本: {tokenizer_config.get('version', '未知')}")
            print(f"  - 模型类型: {tokenizer_config.get('model_type', '未知')}")
        except Exception as e:
            print(f"[错误] 配置文件加载失败: {str(e)}")


    @property
    def is_fast(self) -> bool:
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        return True

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        # print("加密中=========")
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        # print("加密中=========")
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        return self._tokenizer.decoder

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        """
        Convert the encoding representation to python Dict and list of encodings.
        添加详细打印信息跟踪编码转换流程
        """
        print("\n=== 开始编码转换 ===")
        print(f"传入参数：return_overflowing_tokens={return_overflowing_tokens}, return_offsets_mapping={return_offsets_mapping}")

        # 参数初始化验证
        return_token_type_ids = return_token_type_ids if return_token_type_ids is not None else "token_type_ids" in self.model_input_names
        return_attention_mask = return_attention_mask if return_attention_mask is not None else "attention_mask" in self.model_input_names
        print(f"自动设置参数：token_type_ids={return_token_type_ids}, attention_mask={return_attention_mask}")

        # 处理溢出token
        overflow_status = "检测到溢出" if encoding.overflowing else "无溢出"
        print(f"\n溢出检测：{overflow_status}")
        if return_overflowing_tokens and encoding.overflowing:
            encodings = [encoding] + encoding.overflowing
            print(f"当前总编码数：{len(encodings)} (原始+{len(encoding.overflowing)}溢出)")
        else:
            encodings = [encoding]
            print("未启用溢出token返回或无可溢出编码")

        # 构建编码字典
        print("\n构建编码字典：")
        encoding_dict = defaultdict(list)
        for idx, e in enumerate(encodings):
            print(f"\n处理第 {idx+1}/{len(encodings)} 个编码：")
            
            # 核心字段
            encoding_dict["input_ids"].append(e.ids)
            print(f"添加input_ids（长度：{len(e.ids)}）")
            
            # 可选字段
            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
                print(f"添加token_type_ids（长度：{len(e.type_ids)}）")
            
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
                print(f"添加attention_mask（长度：{len(e.attention_mask)}）")
            
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
                print(f"添加special_tokens_mask（长度：{len(e.special_tokens_mask)}）")
            
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
                print(f"添加offset_mapping（首项：{e.offsets[0] if e.offsets else None}）")
            
            if return_length:
                encoding_dict["length"].append(len(e.ids))
                print(f"添加length值：{len(e.ids)}")

        # 最终输出验证
        print("\n转换完成，输出结构：")
        print(f"生成字段列表：{list(encoding_dict.keys())}")
        print(f"总编码数：{len(encodings)}")
        print(f"首编码input_ids长度：{len(encodings[0].ids) if encodings else 0}")
        
        return encoding_dict, encodings


    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a Iterable of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `Iterable[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        # print("加密中=========")
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        # # print("加密中=========")
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        # # print("加密中=========")
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        # # print("加密中=========")
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        print("token类型：", self._tokenizer)
        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        print("进行转换")
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        print("生成口令")
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
        padding_side: Optional[bool],
    ):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy ([`~utils.PaddingStrategy`]):
                The kind of padding that will be applied to the input
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):
                The kind of truncation that will be applied to the input
            max_length (`int`):
                The maximum size of a sequence.
            stride (`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
        """
        # print("加密中=========")
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }

            # _truncation might contain more keys that the target `transformers`
            # supports. Use only the target keys to trigger `enable_truncation`.
            # This should enable this code to works on various `tokenizers`
            # targets.
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}

            if current != target:
                self._tokenizer.enable_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": padding_side if padding_side is not None else self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if _padding != target:
                self._tokenizer.enable_padding(**target)

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
    ) -> BatchEncoding:
        print("进行细节操作")
    # def ok():
        # 输入类型检查
        print(f"开始输入类型检查，输入类型为：{type(batch_text_or_text_pairs)}")
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(
                f"batch_text_or_text_pairs has to be a list or a tuple (got {type(batch_text_or_text_pairs)})"
            )
        print("√ 输入类型检查通过，类型为列表/元组")

        # 设置截断与填充策略
        print("\n开始设置截断与填充策略...")
        print(f"参数详情: padding_strategy={padding_strategy}, truncation_strategy={truncation_strategy}, "
              f"max_length={max_length}, stride={stride}, pad_to_multiple_of={pad_to_multiple_of}")
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )
        print("√ 策略设置完成")

        # 检查特殊token分割设置
        print(f"\n检查特殊token分割: 当前split_special_tokens={split_special_tokens}，"
              f"与当前设置{self._tokenizer.encode_special_tokens}是否一致？")
        if self._tokenizer.encode_special_tokens != split_special_tokens:
            self._tokenizer.encode_special_tokens = split_special_tokens
            print("→ 检测到不一致，已更新split_special_tokens设置")
        else:
            print("→ 设置一致，无需修改")

        # 批量编码过程
        print("\n开始批量编码...")
        print(f"参数详情: add_special_tokens={add_special_tokens}, is_split_into_words={is_split_into_words}")
        print(f"样本数量: {len(batch_text_or_text_pairs)}")
        print("首样本示例:", batch_text_or_text_pairs[0][:50] + "...", "样板长度：", len(batch_text_or_text_pairs[0]))  # 打印首样本前50字符
        
        # print("使用的_tokenizer类：", self._tokenizer)
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )
        print(f"√ 编码完成，共获得{len(encodings)}个编码结果")
        print("首编码结构示例:", type(encodings[0]), "长度:", len(encodings[0]), "前面50个内容：", encodings[0])

        # 编码结果转换
        print("\n开始编码转换...")
        print(f"返回参数: return_token_type_ids={return_token_type_ids}, return_attention_mask={return_attention_mask}")
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]
        print("√ 转换完成")
        print("首元素转换结果示例 - 字典键:", tokens_and_encodings[0][0].keys())
        print("首元素编码信息类型:", type(tokens_and_encodings[0][1]))

        # 数据格式整理
        print("\n开始数据清洗与格式整理...")
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
            print(f"字段 {key} 数据量: {len(stack)}")  # 各字段数据量统计
        
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]
        print(f"清洗后总编码数: {len(sanitized_encodings)}")


        print("\n=== 开始处理溢出token映射 ===")
        
        # 溢出token映射处理
        overflow_to_sample_mapping = []
        if return_overflowing_tokens:
            print("\n生成溢出token映射关系...")
            print(f"总样本数: {len(tokens_and_encodings)}")
            
            for i, (toks, _) in enumerate(tokens_and_encodings):
                print(f"\n正在处理样本 {i}:")
                print("原始token数:", len(toks['input_ids']))
                
                # 获取当前样本溢出次数
                overflow_count = len(toks['input_ids'])  # 每个块视为一次"溢出"
                print(f"溢出次数计算: len(input_ids) = {overflow_count}")
                
                # 生成映射关系
                mapping_segment = [i] * overflow_count
                print(f"添加映射段: {mapping_segment}")
                
                overflow_to_sample_mapping += mapping_segment
                print(f"更新后的映射数组: {overflow_to_sample_mapping[-overflow_count:]}")

                print(f"样本 {i} 处理完成，累计映射数: {len(overflow_to_sample_mapping)}")
            
            print("\n最终映射数组生成:")
            print(f"总映射数: {len(overflow_to_sample_mapping)}")
            print("前10个映射索引:", overflow_to_sample_mapping[:10])
            
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping
            print("\n映射关系已存入sanitized_tokens")
        
        # 长度校验
        print("\n开始序列长度校验...")
        for idx, input_ids in enumerate(sanitized_tokens["input_ids"]):
            print(f"校验序列 {idx}，长度: {len(input_ids)}")
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        
        print("\n生成最终BatchEncoding对象")
        return BatchEncoding(
            sanitized_tokens, 
            sanitized_encodings, 
            tensor_type=return_tensors
        )


    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
    # def ok():
        print("开始执行ok方法，准备处理输入文本/文本对...")
        
        # 构造批量输入数据
        if text_pair:
            print(f"检测到文本对输入，将text和text_pair包装成元组列表。text长度: {len(text)}, text_pair长度: {len(text_pair)}")
            batched_input = [(text, text_pair)]
        else:
            print(f"单文本输入模式，将text包装成单元素列表。text长度: {len(text)}")
            batched_input = [text]
        
        print("\n开始调用核心编码方法_batch_encode_plus，参数详情:")
        print(f"• 填充策略: {padding_strategy.name}")          # 例如 PaddingStrategy.LONGEST
        print(f"• 截断策略: {truncation_strategy.name}")       # 例如 TruncationStrategy.LONGEST_FIRST
        print(f"• 最大长度: {max_length} | 步长: {stride}")
        print(f"• 返回张量类型: {return_tensors}")
        print(f"• 特殊token拆分: {split_special_tokens}")
        
        batched_output = self._batch_encode_plus(
            batched_input,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )
        print("\n_batch_encode_plus执行完成，输出包含以下键:", batched_output.keys())
        
        # print("\n加密中=========")  # 原调试信息保留
        
        # 后处理逻辑
        if return_tensors is None and not return_overflowing_tokens:
            print("\n进入后处理分支：return_tensors=None且不需要溢出token")
            print("正在移除批次维度，检查首元素类型...")
            
            processed_data = {
                key: (value[0] if len(value) > 0 and isinstance(value[0], list) else value)
                for key, value in batched_output.items()
            }
            print("维度处理后数据样例:", {k: v[:1] for k, v in processed_data.items()})  # 显示部分数据
            
            batched_output = BatchEncoding(processed_data, batched_output.encodings)
            print("已重新封装为BatchEncoding对象")
        else:
            print("\n跳过维度处理：需要返回张量或包含溢出token")
        
        # 长度警告检查
        print("\n正在检查输入序列长度是否超过模型限制...")
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)
        
        print("\n处理完成，返回最终编码结果")
        return batched_output


    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return (
            self.backend_tokenizer.decoder.decode(tokens)
            if self.backend_tokenizer.decoder is not None
            else " ".join(tokens)
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens as well as in a unique JSON
        file containing {config + vocab + added-tokens}.
        """
        save_directory = str(save_directory)

        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "Your tokenizer does not have a legacy version defined and therefore cannot register this version. You"
                " might consider leaving the legacy_format at `None` or setting it to `False`."
            )

        save_slow = (
            (legacy_format is None or legacy_format is True)
            and self.slow_tokenizer_class is not None
            and self.can_save_slow_tokenizer
        )
        save_fast = legacy_format is None or legacy_format is False

        if save_slow:
            added_tokens_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
            )
            # make sure to be foward compatible
            added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
            if added_vocab:
                with open(added_tokens_file, "w", encoding="utf-8") as f:
                    out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                    f.write(out_str)

            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)

        if save_fast:
            tokenizer_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
            )
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)

        return file_names

    def train_new_from_iterator(
        self,
        text_iterator,
        vocab_size,
        length=None,
        new_special_tokens=None,
        special_tokens_map=None,
        **kwargs,
    ):
        """
        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
        as the current one.

        Args:
            text_iterator (generator of `List[str]`):
                The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
                if you have everything in memory.
            vocab_size (`int`):
                The size of the vocabulary you want for your tokenizer.
            length (`int`, *optional*):
                The total number of sequences in the iterator. This is used to provide meaningful progress tracking
            new_special_tokens (list of `str` or `AddedToken`, *optional*):
                A list of new special tokens to add to the tokenizer you are training.
            special_tokens_map (`Dict[str, str]`, *optional*):
                If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
                token name to new special token name in this argument.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

        Returns:
            [`PreTrainedTokenizerFast`]: A new tokenizer of the same type as the original one, trained on
            `text_iterator`.

        """
        tokenizer_json = json.loads(self._tokenizer.to_str())
        # Remove added tokens for now (uses IDs of tokens)
        added_tokens = tokenizer_json.pop("added_tokens")
        # Remove post processor for now (uses IDs of tokens)
        post_processor = tokenizer_json.pop("post_processor")

        unk_token = None
        # Remove vocab
        if tokenizer_json["model"]["type"] == "BPE":
            tokenizer_json["model"]["vocab"] = {}
            tokenizer_json["model"]["merges"] = []
        elif tokenizer_json["model"]["type"] == "Unigram":
            if tokenizer_json["model"]["unk_id"] is not None:
                unk_id = tokenizer_json["model"]["unk_id"]
                unk_token = tokenizer_json["model"]["vocab"][unk_id][0]
                if special_tokens_map is not None and unk_token in special_tokens_map:
                    unk_token = special_tokens_map[unk_token]
                tokenizer_json["model"]["unk_id"] = 0
                tokenizer_json["model"]["vocab"] = [[unk_token, 0.0]]
        elif tokenizer_json["model"]["type"] in ["WordLevel", "WordPiece"]:
            tokenizer_json["model"]["vocab"] = {}
        else:
            raise ValueError(
                f"This method does not support this type of tokenizer (found {tokenizer_json['model']['type']}) "
                "only BPE, Unigram, WordLevel and WordPiece."
            )

        if (
            special_tokens_map is not None
            and "unk_token" in tokenizer_json["model"]
            and tokenizer_json["model"]["unk_token"] in special_tokens_map
        ):
            tokenizer_json["model"]["unk_token"] = special_tokens_map[tokenizer_json["model"]["unk_token"]]

        tokenizer = TokenizerFast.from_str(json.dumps(tokenizer_json))

        # Get the special tokens from the current tokenizer if none are specified.
        special_tokens = []
        for added_token in added_tokens:
            special = added_token.pop("special", None)
            _ = added_token.pop("id", None)
            if tokenizer_json["model"]["type"] != "Unigram" and not special:
                continue
            if special_tokens_map is not None and added_token["content"] in special_tokens_map:
                added_token["content"] = special_tokens_map[added_token["content"]]
            special_tokens.append(AddedToken(**added_token))

        if new_special_tokens is not None:
            special_tokens.extend(new_special_tokens)

        # Trainer needs to know the end of word / continuing subword thingies in BPE
        if (
            tokenizer_json["model"]["type"] == "BPE"
            and "continuing_subword_prefix" not in kwargs
            and tokenizer_json["model"]["continuing_subword_prefix"] is not None
        ):
            kwargs["continuing_subword_prefix"] = tokenizer_json["model"]["continuing_subword_prefix"]
        if (
            tokenizer_json["model"]["type"] == "BPE"
            and "end_of_word_suffix" not in kwargs
            and tokenizer_json["model"]["end_of_word_suffix"] is not None
        ):
            kwargs["end_of_word_suffix"] = tokenizer_json["model"]["end_of_word_suffix"]
        if tokenizer_json["model"]["type"] == "Unigram" and unk_token is not None:
            kwargs["unk_token"] = unk_token
        if tokenizer_json["pre_tokenizer"] is not None:
            if (
                tokenizer_json["pre_tokenizer"]["type"] == "ByteLevel"
                or tokenizer_json["pre_tokenizer"]["type"] == "Sequence"
                and "pretokenizers" in tokenizer_json["pre_tokenizer"]
                and any(
                    pretokenizer["type"] == "ByteLevel"
                    for pretokenizer in tokenizer_json["pre_tokenizer"]["pretokenizers"]
                )
            ):
                kwargs["initial_alphabet"] = pre_tokenizers_fast.ByteLevel.alphabet()

        trainer_class = MODEL_TO_TRAINER_MAPPING[tokenizer_json["model"]["type"]]
        trainer = trainer_class(vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
        tokenizer.train_from_iterator(text_iterator, length=length, trainer=trainer)

        if post_processor is not None:
            trained_tokenizer_json = json.loads(tokenizer.to_str())
            # Almost done, we just have to adjust the token IDs in the post processor
            if "special_tokens" in post_processor:
                for key in post_processor["special_tokens"]:
                    tokens = post_processor["special_tokens"][key]["tokens"]
                    if special_tokens_map is not None:
                        tokens = [special_tokens_map.get(token, token) for token in tokens]
                    post_processor["special_tokens"][key]["tokens"] = tokens
                    for token in tokens:
                        token_id = tokenizer.token_to_id(token)
                        if token_id is None:
                            raise ValueError(
                                "Attempted to set a token in the post processor that does not exist in the mapping"
                            )

                    post_processor["special_tokens"][key]["ids"] = [tokenizer.token_to_id(token) for token in tokens]

            for special_token in ["cls", "sep"]:
                if special_token in post_processor:
                    token, _ = post_processor[special_token]
                    if special_tokens_map is not None and token in special_tokens_map:
                        token = special_tokens_map[token]
                    token_id = tokenizer.token_to_id(token)
                    if token_id is None:
                        raise ValueError(
                            "Attempted to set a token in the post processor that does not exist in the mapping"
                        )
                    post_processor[special_token] = [token, token_id]

            trained_tokenizer_json["post_processor"] = post_processor
            tokenizer = TokenizerFast.from_str(json.dumps(trained_tokenizer_json))

        kwargs = self.init_kwargs.copy()
        # Map pad/cls/mask token at the Transformers level
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove("additional_special_tokens")
        for token in special_tokens_list:
            if getattr(self, token) is not None:
                special_token = getattr(self, token)
                if special_tokens_map is not None and special_token in special_tokens_map:
                    special_token = special_tokens_map[special_token]

                special_token_full = self._special_tokens_map.get(token, None)
                if isinstance(special_token_full, AddedToken):
                    # Create an added token with the same parameters except the content
                    kwargs[token] = AddedToken(
                        special_token,
                        single_word=special_token_full.single_word,
                        lstrip=special_token_full.lstrip,
                        rstrip=special_token_full.rstrip,
                        normalized=special_token_full.normalized,
                        special=True,
                    )
                else:
                    kwargs[token] = special_token

        additional_special_tokens = self.additional_special_tokens
        if new_special_tokens is not None:
            additional_special_tokens.extend(new_special_tokens)
        if len(additional_special_tokens) > 0:
            kwargs["additional_special_tokens"] = additional_special_tokens

        return self.__class__(tokenizer_object=tokenizer, **kwargs)
