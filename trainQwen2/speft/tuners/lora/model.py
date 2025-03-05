# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import operator
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from speft.import_utils import is_bnb_4bit_available, is_bnb_available
from speft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from speft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from speft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from speft.utils.other import get_pattern_key

from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .eetq import dispatch_eetq
from .gptq import dispatch_gptq
from .hqq import dispatch_hqq
from .layer import Conv2d, LoraLayer, dispatch_default
from .torchao import dispatch_torchao
from .tp_layer import dispatch_megatron


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class LoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from speft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from speft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )
        >>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     bos_token="[BOS]",
        ...     eos_token="[EOS]",
        ...     unk_token="[UNK]",
        ...     pad_token="[PAD]",
        ...     mask_token="[MASK]",
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     quantization_config=quantization_config,
        ... )
        >>> model = prepare_model_for_kbit_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _prepare_model(self, peft_config: LoraConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        if peft_config.layer_replication:
            replicate_layers(model, peft_config.layer_replication)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
    #  def ok32432():
        print("\n🚀 开始配置LoRA层参数")
        print(f"📌 当前处理模块路径: {current_key or '未指定'}")

        # 空键检查
        if current_key is None:
            print("❌ 致命错误: current_key参数未传入")
            print("🛑 可能原因:")
            print("1. 模块路径生成逻辑错误")
            print("2. 模型结构解析异常")
            raise ValueError("Current Key shouldn't be `None`")

        # 正则匹配获取配置键
        print("\n🔍 执行正则模式匹配:")
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        print(f"   🎯 rank模式键: {r_key} (可用模式: {list(lora_config.rank_pattern.keys())})")
        print(f"   α模式键: {alpha_key} (可用模式: {list(lora_config.alpha_pattern.keys())})")

        # 获取动态rank/alpha值
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)
        print(f"\n📊 动态参数计算:")
        print(f"   🔢 rank = {r} ({'默认' if r_key not in lora_config.rank_pattern else '模式匹配'})")
        print(f"   α系数 = {alpha} ({'默认' if alpha_key not in lora_config.alpha_pattern else '模式匹配'})")

        # 准备基础参数
        print("\n📦 组装LoRA配置参数:")
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": lora_config.lora_bias,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        print("✅ 基础参数配置完成:")
        for k, v in kwargs.items():
            print(f"   {k:20} = {v}")

        # 处理Tensor子类配置
        print("\n🔧 配置Tensor子类方法:")
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
            print(f"✅ 成功获取Tensor子类应用方法: {kwargs['get_apply_tensor_subclass']}")
        except AttributeError:
            print("⚠️ 未找到hf_quantizer配置，跳过Tensor子类设置")

        # 量化配置扫描
        print("\n📡 扫描量化配置:")
        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            print(f"   🔎 检查{quant_method.upper()}量化...", end="")
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config
                print(f"✅ 检测到配置 | 版本: {quantization_config.__class__.__name__}")
            else:
                print("❌ 未找到")


    # def ok3543():
        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from speft.tuners.adalora import AdaLoraLayer

    # def ok3543():
        print("\n🚀 开始执行模块更新/替换流程")
        print(f"📌 当前适配器: {adapter_name}")
        print(f"🔧 目标模块类型: {type(target).__name__}")

        # 检查模块类型条件
        is_lora_layer = isinstance(target, LoraLayer)
        is_adalora = isinstance(target, AdaLoraLayer)
        print(f"\n🔍 模块类型验证 - LoraLayer: {is_lora_layer} | 非AdaLora: {not is_adalora}")
        
        if is_lora_layer and not is_adalora:
            print("✅ 符合层更新条件，执行参数更新")
            print(f"🔄 更新参数列表: [r={r}, alpha={alpha}, dropout={lora_config.lora_dropout}]")
            print(f"⚙️ 初始化方式: {lora_config.init_lora_weights}")
            print(f"🌀 高级选项: use_rslora={lora_config.use_rslora}, use_dora={lora_config.use_dora}")

            # 记录更新前参数状态
            pre_params = sum(p.numel() for p in target.parameters())
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                lora_bias=lora_config.lora_bias,
            )
            # 更新后参数对比
            post_params = sum(p.numel() for p in target.parameters())
            print(f"📊 参数量变化: {pre_params} → {post_params} (+{post_params - pre_params})")
        else:
            print("🛠️ 需要创建新模块替换原模块")
            print(f"📦 使用配置创建新模块: { {k:v for k,v in kwargs.items() if not k.endswith('_config')} }")
            
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            print(f"🎯 新模块类型: {type(new_module).__name__}")

            # 检查激活状态
            print(f"\n🔍 检查适配器激活状态: {adapter_name} in {self.active_adapters}")
            if adapter_name not in self.active_adapters:
                print("⚠️ 新适配器未激活，冻结参数")
                new_module.requires_grad_(False)
                frozen_params = sum(p.numel() for p in new_module.parameters())
                print(f"❄️ 冻结参数数量: {frozen_params}")
            else:
                print(f"可以激活的列表合集{self.active_adapters}")
                print("✅ 保持新模块可训练状态")

            # 执行模块替换
            print(f"\n🔄 开始模块替换操作")
            print(f"父模块: {parent.__class__.__name__}")
            print(f"目标属性名: {target_name}")
            print(f"原模块类型: {type(target).__name__} → 新类型: {type(new_module).__name__}")

            self._replace_module(parent, target_name, new_module, target)
            
            # 替换后验证
            replaced_module = getattr(parent, target_name, None)
            print(f"🔍 替换验证: {replaced_module is not None and replaced_module == new_module}")
            print(f"📌 新模块ID: {id(replaced_module)}")

        print("\n🎉 模块处理完成")
        print("="*60)


    def _replace_module(self, parent, child_name, new_module, child):
    #  def ok23432():
        print("\n🚀 开始执行模块替换后处理流程")
        print(f"📌 父模块类型: {type(parent).__name__}")
        print(f"🔧 替换操作: {child_name} → {type(new_module).__name__}")

        # 执行模块替换
        print(f"\n🔄 设置父模块属性 [{child_name}]")
        setattr(parent, child_name, new_module)
        print(f"✅ 验证替换结果: {hasattr(parent, child_name)}")

        # 解包基础层
        original_child = child
        if hasattr(child, "base_layer"):
            print("\n🎁 检测到基础层封装，解包操作:")
            child = child.base_layer
            print(f"   📦 原模块类型: {type(original_child).__name__}")
            print(f"   🎯 新基础层类型: {type(child).__name__}")

        # 权重转移逻辑
        print("\n🏋️ 开始权重/偏置转移:")
        if not hasattr(new_module, "base_layer"):
            weight_source = None
            # HQQ特殊处理
            if hasattr(new_module, "W_q"):
                print("🎯 检测到HQQ量化格式 (W_q)")
                new_module.W_q = child.W_q
                weight_source = "W_q"
            else:
                print("⚖️ 标准权重转移")
                new_module.weight = child.weight
                weight_source = "weight"

            print(f"   📥 权重来源: {type(child).__name__}.{weight_source}")
            print(f"   📊 权重形状: {getattr(child, weight_source).shape}")

            if hasattr(child, "bias"):
                print("⚖️ 偏置转移")
                new_module.bias = child.bias
                print(f"   📐 偏置形状: {child.bias.shape}")
            else:
                print("⚠️ 未检测到偏置参数")

        # 状态转移
        print("\n📦 处理模块状态:")
        if getattr(child, "state", None) is not None:
            print(f"✅ 检测到状态参数 (keys: {child.state.keys()})")
            
            if hasattr(new_module, "base_layer"):
                print("   🎯 转移到base_layer")
                new_module.base_layer.state = child.state
            else:
                print("   🎯 直接设置状态")
                new_module.state = child.state
            
            print(f"🚚 设备迁移: {child.weight.device}")
            new_module.to(child.weight.device)
            print(f"✅ 新模块设备: {next(new_module.parameters()).device}")
        else:
            print("⚠️ 无状态需要转移")

        # 设备分配
        print("\n🔧 检查元设备分配:")
        meta = torch.device("meta")
        for name, module in new_module.named_modules():
            if self.prefix in name or "ranknum" in name:
                print(f"\n🔍 处理子模块: {name}")
                
                # 确定权重源
                weight = None
                if hasattr(child, "qweight"):
                    weight = child.qweight
                    src_desc = "qweight(量化)"
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                    src_desc = "W_q(HQQ)"
                elif hasattr(child, "weight"):
                    weight = child.weight
                    src_desc = "weight"
                else:
                    weight = next(child.parameters())
                    src_desc = "首个参数"
                
                print(f"   ⚖️ 权重源: {src_desc} | 设备: {weight.device}")
                
                # 检查元设备
                if not any(p.device == meta for p in module.parameters()):
                    print(f"   🚚 迁移到 {weight.device}")
                    module.to(weight.device)
                    # print(f"   ✅ 当前设备: {next(module.parameters()).device}")
                else:
                    print("   ⚠️ 检测到元设备，保持延迟加载状态")

        print("\n🎉 模块后处理完成")
        print("="*60)


    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
    # def ok54354353():
        print("\n🚀 开始构建LoRA分派器列表")
        dispatchers = []
        print(f"📦 初始分派器列表: {len(dispatchers)} 个")

        # 处理自定义模块配置
        if lora_config._custom_modules:
            print("\n🔧 检测到自定义模块配置")
            print(f"📋 自定义映射数量: {len(lora_config._custom_modules)}")
            print(f"🔍 自定义类型示例: {list(lora_config._custom_modules.keys())[:3]}...")

            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                print(f"\n⚡ 执行动态分派 | 目标: {type(target).__name__}")
                new_module = None
                
                # 获取基础层
                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                    print(f"   🎯 基础层类型: {type(target_base_layer).__name__} (来自TunerLayer)")
                else:
                    target_base_layer = target
                    print(f"   🎯 基础层类型: {type(target_base_layer).__name__}")

                # 遍历自定义映射
                for idx, (key, custom_cls) in enumerate(lora_config._custom_modules.items()):
                    print(f"   🔄 尝试匹配 [{idx+1}/{len(lora_config._custom_modules)}] {key}...", end="")
                    if isinstance(target_base_layer, key):
                        print("✅ 匹配成功")
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        print(f"   🛠️ 创建 {custom_cls.__name__} 实例 | 参数: {kwargs.keys()}")
                        break
                    else:
                        print("❌ 类型不匹配")

                if new_module:
                    print(f"🎯 成功创建自定义模块: {type(new_module).__name__}")
                else:
                    print("⚠️ 警告: 未找到匹配的自定义模块")
                return new_module

            dispatchers.append(dynamic_dispatch_func)
            print(f"\n✅ 已添加动态分派器 | 当前分派器数: {len(dispatchers)}")

        # 处理bnb 8bit
        print("\n🔌 检查bitsandbytes 8bit可用性")
        if is_bnb_available():
            print("✅ 检测到bitsandbytes 8bit")
            from .bnb import dispatch_bnb_8bit
            print(f"📥 导入dispatch_bnb_8bit ({dispatch_bnb_8bit.__module__})")
            dispatchers.append(dispatch_bnb_8bit)
            print(f"📌 当前分派器数: {len(dispatchers)}")
        else:
            print("⚠️ 未检测到bitsandbytes 8bit安装")

        # 处理bnb 4bit
        print("\n🔌 检查bitsandbytes 4bit可用性")
        if is_bnb_4bit_available():
            print("✅ 检测到bitsandbytes 4bit")
            from .bnb import dispatch_bnb_4bit
            print(f"📥 导入dispatch_bnb_4bit ({dispatch_bnb_4bit.__module__})")
            dispatchers.append(dispatch_bnb_4bit)
            print(f"📌 当前分派器数: {len(dispatchers)}")
        else:
            print("⚠️ 未检测到bitsandbytes 4bit安装")


    # def ok232():
        print("\n🚀 开始执行模块分派流程")
        print(f"📌 目标模块: {target.__class__.__name__}")
        print(f"🔧 适配器名称: {adapter_name}")
        print(f"⚙️ LoRA配置参数: r={lora_config.r}, alpha={lora_config.lora_alpha}")

        # 扩展分派器列表
        print("\n📦 加载标准分派器集合:")
        standard_dispatchers = [
            dispatch_eetq, dispatch_aqlm, dispatch_awq, 
            dispatch_gptq, dispatch_hqq, dispatch_torchao,
            dispatch_megatron, dispatch_default
        ]
        dispatchers.extend(standard_dispatchers)
        print(f"✅ 已添加 {len(standard_dispatchers)} 个标准分派器:")
        for i, d in enumerate(standard_dispatchers, 1):
            print(f"   [{i}] {d.__name__ if callable(d) else '无名分派器'}")
        print(f"📌 总分派器数量: {len(dispatchers)}")

        new_module = None
        print("\n🔍 开始尝试分派器链式匹配:")
        for idx, dispatcher in enumerate(dispatchers, 1):
            dispatcher_name = dispatcher.__name__ if callable(dispatcher) else str(dispatcher)
            print(f"\n⚡ 尝试分派器 [{idx}/{len(dispatchers)}] {dispatcher_name}")
            
            try:
                new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
                print(f"   🧪 执行结果: {'成功' if new_module else '未匹配'}") 
                
                if new_module:
                    print(f"🎯 {dispatcher_name} 分派成功！")
                    print(f"📦 创建模块类型: {new_module.__class__.__name__}")
                    print(f"🔗 模块ID: {id(new_module)}")
                    break
            except Exception as e:
                print(f"⚠️ 分派器异常: {str(e)}")
                continue

        # 错误处理
        if new_module is None:
            print("\n❌ 所有分派器尝试失败！")
            print("🛑 失败原因分析:")
            print(f"   目标模块类型: {target.__class__.__name__}")
            print(f"   支持的类型列表:")
            print("   - torch.nn.Linear")
            print("   - torch.nn.Embedding") 
            print("   - torch.nn.Conv2d/3d")
            print("   - transformers.Conv1D")
            
            print("\n🔧 建议排查步骤:")
            print("1. 检查目标模块是否继承自支持的基础类型")
            print("2. 验证是否加载了必要的量化依赖（如bnb、gptq等）")
            print("3. 尝试添加自定义模块分派器")
            
            raise ValueError(
                f"Unsupported module type {target.__class__.__name__}. "
                "See console logs for supported types and debugging tips."
            )

        print("\n🎉 模块分派流程完成")
        print("="*60)


        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        # Check that users only passed actually existing adapters.
        # Note: We cannot do this on the layer level, as each individual layer may not have each adapter. Still, we want
        # to check that there is at least one layer with the given name, or else something like typos can easily slip.
        expected_adapters = set()
        for layer in self.modules():
            if isinstance(layer, LoraLayer):
                expected_adapters |= layer.lora_A.keys()
                expected_adapters |= layer.lora_embedding_A.keys()
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        unexpected_adapters = unique_adapters - expected_adapters
        if unexpected_adapters:
            raise ValueError(f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, LoraLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        super()._check_merge_allowed()
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge LORA layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge LORA layers when base model layers are replicated")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def _check_add_weighted_adapter(
        self, adapters: list[str], combination_type: str, svd_rank: int | None
    ) -> tuple[str, int, str]:
        """
        Helper function to check if the arguments to add_weighted_adapter are valid and compatible with the underlying
        model.
        """
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # If more than one of the adapters targets the same module with modules_to_save, raise an error, as these
        # modules cannot be merged. First, find the ModulesToSaveWrapper instances in the model, then check if they
        # have modules for the adapters to be merged.
        modules_to_save_wrappers = [module for module in self.modules() if isinstance(module, ModulesToSaveWrapper)]
        problematic_wrappers = [
            wrapper
            for wrapper in modules_to_save_wrappers
            if sum(adapter in wrapper.modules_to_save for adapter in adapters) > 1
        ]
        if problematic_wrappers:
            raise ValueError(
                "Cannot add weighted adapters if they target the same module with modules_to_save, but found "
                f"{len(problematic_wrappers)} such instance(s)."
            )

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or "
                    "dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] is str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] is set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        return combination_type, new_rank, new_target_modules

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
        svd_rank: int | None = None,
        svd_clamp: int | None = None,
        svd_full_matrices: bool = True,
        svd_driver: str | None = None,
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """

        if adapter_name in list(self.peft_config.keys()):
            return

        combination_type, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type=combination_type,
            svd_rank=svd_rank,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
                    loras_A = torch.cat(loras_A, dim=0)
                    loras_B = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        valid_weights = []
        is_embedding = any(adapter in target.lora_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
        if combination_type == "svd":
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        # account weights for LoRA A and B layers.
        valid_weights = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)
        valid_weights = torch.tensor(valid_weights).to(lora_A_deltas[0].device)
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype
        for i, task_tensors in enumerate(lora_deltas):
            if combination_type == "linear":
                lora_deltas[i] = task_arithmetic(task_tensors, valid_weights)
            elif combination_type == "ties":
                lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "dare_linear":
                lora_deltas[i] = dare_linear(task_tensors, valid_weights, density)
            elif combination_type == "dare_ties":
                lora_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                lora_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
            else:
                raise ValueError("Invalid combination type")
        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from speft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def subtract_mutated_init(self, output_state_dict: dict[str, torch.Tensor], adapter_name: str, kwargs=None):
        """
        This function can calculate the updates of the [PiSSA | OLoRA] by comparing the parameters of the [PiSSA |
        OLoRA] adapter in `output_state_dict` with the initial values of [PiSSA | OLoRA] in `adapter_name`, thus
        converting [PiSSA | OLoRA] to LoRA.
        """
        for name, param in self.model.named_parameters():
            if (
                param.data.dtype != torch.float32
                and param.data.dtype != torch.float16
                and param.data.dtype != torch.bfloat16
            ) and adapter_name.startswith("pissa"):
                warnings.warn(
                    r"Note that Quant(W_res) + AB != Quant(W) + \Delta(AB); "
                    "the converted LoRA, when combined with W or Quant(W), may introduce a certain gap in the fine-tuned model. "
                    "Therefore, we recommend directly using the Quant(W_res) in conjunction with the PiSSA adapter. "
                )
        mutated_init_state_dict = get_peft_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
            adapter_name=adapter_name,
        )
        tensors_lora = {}
        for name in output_state_dict.keys():
            ## W = W^{res} + A_0 \times B_0,
            ## W + \Delta W = W^{res} + A \times B,
            ## \Delta W = A \times B - A_0 \times B_0 = [A | A_0] \times [B | -B_0]^T = A'B'.
            if "lora_A" in name:
                tensors_lora[name] = torch.cat(
                    [output_state_dict[name], mutated_init_state_dict[".".join(name.split(".")[1:])]], dim=0
                )
            elif "lora_B" in name:
                tensors_lora[name] = torch.cat(
                    [output_state_dict[name], -mutated_init_state_dict[".".join(name.split(".")[1:])]], dim=1
                )

        return tensors_lora
