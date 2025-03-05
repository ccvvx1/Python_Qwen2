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
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from speft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from speft.utils.integrations import dequantize_module_weight, gather_params_ctx, get_bnb_param_type
from speft.utils.other import transpose

from .config import LoraConfig
from .dora import DoraConv2dLayer, DoraConv3dLayer, DoraEmbeddingLayer, DoraLinearLayer, _DoraConvNdLayer


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
    # def ok234324():
        print("开始初始化LoRA层...")
        self.base_layer = base_layer
        print(f"设置基础层: {type(self.base_layer).__name__}")

        self.r = {}
        print("初始化秩(r)字典: 空字典")

        self.lora_alpha = {}
        print("初始化缩放系数(lora_alpha)字典: 空字典")

        self.scaling = {}
        print("初始化最终缩放因子(scaling)字典: 空字典")

        self.lora_dropout = nn.ModuleDict({})
        print("创建LoRA Dropout模块字典: ModuleDict初始化完成")

        self.lora_A = nn.ModuleDict({})
        print("创建LoRA_A矩阵模块字典: ModuleDict初始化完成")

        self.lora_B = nn.ModuleDict({})
        print("创建LoRA_B矩阵模块字典: ModuleDict初始化完成")

        # For Embedding layer
        print("\n初始化嵌入层相关参数...")
        self.lora_embedding_A = nn.ParameterDict({})
        print("创建嵌入层LoRA_A参数字典: ParameterDict初始化完成")

        self.lora_embedding_B = nn.ParameterDict({})
        print("创建嵌入层LoRA_B参数字典: ParameterDict初始化完成")

        # Mark the weight as unmerged
        print("\n设置适配器状态...")
        self._disable_adapters = False
        print(f"禁用适配器标志: {self._disable_adapters}")

        self.merged_adapters = []
        print(f"初始化已合并适配器列表: {self.merged_adapters}")

        self.use_dora: dict[str, bool] = {}
        print("初始化DoRA使用标记字典: 空字典")

        self.lora_bias: dict[str, bool] = {}
        print("初始化LoRA偏置标记字典: 空字典")

        self.lora_magnitude_vector = torch.nn.ModuleDict()
        print("创建DoRA模长向量模块字典: ModuleDict初始化完成")

        self._caches: dict[str, Any] = {}
        print("初始化缓存字典: 空字典")

        self.ephemeral_gpu_offload = ephemeral_gpu_offload
        print(f"设置临时GPU卸载标志: {self.ephemeral_gpu_offload}")

        self.kwargs = kwargs
        print(f"接收额外参数: {kwargs.keys() if kwargs else '无'}")

        print("\nLoRA层初始化完成\n")


    # def ok234():
        print("\n开始获取基础层特征维度...")
        base_layer = self.get_base_layer()
        print(f"当前基础层类型: {type(base_layer).__name__}")

        if isinstance(base_layer, nn.Linear):
            print("├─ 处理Linear层")
            in_features, out_features = base_layer.in_features, base_layer.out_features
            print(f"└─ 获取特征: in_features={in_features}, out_features={out_features}")
            
        elif isinstance(base_layer, nn.Conv2d):
            print("├─ 处理2D卷积层")
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
            print(f"└─ 获取通道: in_channels={in_features}, out_channels={out_features}")
            
        elif isinstance(base_layer, nn.Conv3d):
            print("├─ 处理3D卷积层")
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
            print(f"└─ 获取通道: in_channels={in_features}, out_channels={out_features}")
            
        elif isinstance(base_layer, nn.Embedding):
            print("├─ 处理嵌入层")
            in_features = base_layer.num_embeddings
            out_features = base_layer.embedding_dim
            print(f"└─ 获取参数: num_embeddings={in_features}, embedding_dim={out_features}")
            
        elif isinstance(base_layer, Conv1D):
            print("├─ 处理1D卷积层(特殊类型)")
            if hasattr(base_layer.weight, "ds_shape"):
                print("│  └─ 检测到分布式张量(ds_shape)")
                in_features, out_features = base_layer.weight.ds_shape
            else:
                print("│  └─ 使用标准权重形状")
                in_features, out_features = base_layer.weight.shape
            print(f"└─ 最终维度: in={in_features}, out={out_features}")
            
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            print("├─ 检测到QuantLinear量化层")
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
            print(f"└─ 获取量化特征: infeatures={in_features}, outfeatures={out_features}")
            
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            print("├─ 检测到Megatron并行层")
            in_features, out_features = base_layer.input_size, base_layer.output_size
            print(f"└─ 获取并行参数: input_size={in_features}, output_size={out_features}")
            
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            print("├─ 检测到AQLM量化层")
            in_features, out_features = base_layer.in_features, base_layer.out_features
            print(f"└─ 获取AQLM特征: in_features={in_features}, out_features={out_features}")

    # def ok2342():
        print("\n╔══════════════════════════════════════════╗")
        print("║ 开始处理量化层特征提取                 ║")
        print("╚══════════════════════════════════════════╝")
        
        # AWQ量化层检测
        if hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            print("├─ [AWQ] 检测到GEMM量化层")
            in_features, out_features = base_layer.in_features, base_layer.out_features
            print(f"└─ 获取AWQ特征: in={in_features}, out={out_features}")
            
        # Eetq量化层检测    
        elif base_layer.__class__.__name__ == "EetqLinear":
            print("├─ [Eetq] 检测到Eetq量化层")
            in_features, out_features = base_layer.in_features, base_layer.out_features
            print(f"└─ 获取Eetq特征: in={in_features}, out={out_features}")
            
        # HQQ量化层检测
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            print("├─ [HQQ] 检测到HQQ量化层")
            in_features, out_features = base_layer.in_features, base_layer.out_features
            print(f"└─ 获取HQQ特征: in={in_features}, out={out_features}")
            
        else:
            print("├─ 尝试通用层类型处理")
            # 自定义层处理逻辑
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                print("│ ├─ 检测到标准特征属性")
                in_features, out_features = base_layer.in_features, base_layer.out_features
                print(f"│ └─ 获取通用特征: in={in_features}, out={out_features}")
            else:
                print("│ ╠═▶ 警告！未找到标准特征属性")
                in_features, out_features = None, None
                print("│ ╚═▶ 特征维度设置为: (None, None)")
                
            # 发出警告
            warning_msg = f"不支持的层类型: {type(base_layer)}"
            print("╔══════════════════════════════════════════╗")
            print(f"║ ⚠️ 警告: {warning_msg:<25} ║")
            print("╚══════════════════════════════════════════╝")
            warnings.warn(warning_msg, UserWarning)

        # 最终赋值
        print("\n╭────────────────────────────────────────────╮")
        print(f"│ 最终设置: in_features={str(in_features):<5} │ out_features={str(out_features):<5} │")
        print("╰────────────────────────────────────────────╯")
        
        self.in_features = in_features
        self.out_features = out_features


    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
    # def ok43243():
        print("\n▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
        print(f"⚙️ 开始配置适配器 '{adapter_name}'")
        print(f"▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")

        # 参数验证
        print(f"\n🔍 参数验证:")
        print(f"| {'参数':<15} | {'值':<8} | {'类型':<12} |")
        print("|----------------|---------|------------|")
        print(f"| r             | {r:<8} | {type(r).__name__:<12} |")
        print(f"| lora_alpha    | {lora_alpha:<8} | {type(lora_alpha).__name__:<12} |")
        print(f"| lora_dropout  | {lora_dropout:<8.2f} | float       |")
        print(f"| use_rslora    | {use_rslora!s:<8} | {type(use_rslora).__name__:<12} |")
        
        if r <= 0:
            error_msg = f"无效的秩r: {r} (必须为正整数)"
            print("\n❌ 错误:", error_msg)
            raise ValueError(error_msg)
        else:
            print("\n✅ 参数验证通过")

        # 存储基础参数
        print("\n📦 存储适配器参数:")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        print(f"| 参数存储       | r={r}, alpha={lora_alpha} |")

        # Dropout配置
        print("\n🌧️ 配置Dropout层:")
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
            print(f"创建Dropout层 (p={lora_dropout:.2f})")
        else:
            lora_dropout_layer = nn.Identity()
            print("禁用Dropout，使用Identity层")
        
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        print(f"已更新Dropout字典: {list(self.lora_dropout.keys())}")

        # 初始化LoRA矩阵
        print("\n🧮 初始化低秩矩阵:")
        print(f"LoRA_A: in_features={self.in_features} → r={r}")
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        
        print(f"LoRA_B: r={r} → out_features={self.out_features} (bias={lora_bias})")
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias
        print("使用的偏差详情：", lora_bias)

        # 计算缩放因子
        print("\n⚖️ 计算缩放因子:")
        if use_rslora:
            formula = f"{lora_alpha}/√{r} ≈ {lora_alpha/math.sqrt(r):.4f}"
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
            print(f"使用RSLoRA公式: {formula}")
        else:
            formula = f"{lora_alpha}/{r} = {lora_alpha/r:.4f}"
            self.scaling[adapter_name] = lora_alpha / r
            print(f"使用标准公式: {formula}")
        

    # def ok234322():
        print("\n▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜")
        print("🔧 开始适配器初始化流程")
        print(f"▙ 适配器名称: {adapter_name} | 初始化方法: {init_lora_weights} ▟")
        
        # 初始化方法选择
        init_method = str(init_lora_weights).lower()
        print(f"\n🔎 初始化方法检测: {init_method}")
        
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            print(f"├─ [PiSSA] 检测到参数高效初始化方法: {init_lora_weights}")
            with gather_params_ctx(self.get_base_layer().weight):
                print(f"│  ├─ 进入参数收集上下文 (设备: {self.get_base_layer().weight.device})")
                self.pissa_init(adapter_name, init_lora_weights)
                print(f"└─ PiSSA初始化完成，版本: {init_lora_weights.split('_')[-1]}")
                
        elif isinstance(init_lora_weights, str) and init_method == "olora":
            print("├─ [OLoRA] 检测到正交初始化方法")
            with gather_params_ctx(self.get_base_layer().weight):
                print(f"│  ├─ 进入参数收集上下文 (设备: {self.get_base_layer().weight.device})")
                self.olora_init(adapter_name)
                print("└─ OLoRA正交初始化完成")
                
        elif init_method == "loftq":
            print("├─ [LoFTQ] 检测到低精度浮点量化初始化")
            with gather_params_ctx(self.get_base_layer().weight):
                print(f"│  ├─ 进入参数收集上下文 (设备: {self.get_base_layer().weight.device})")
                self.loftq_init(adapter_name)
                print("└─ LoFTQ量化初始化完成")
                
        elif init_method == "eva":
            print("├─ [EVA] 零初始化方法检测")
            print(f"│  └─ 初始化LoRA_B权重为全零 (shape: {self.lora_B[adapter_name].weight.shape})")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            print("└─ EVA零初始化完成")
            
        elif init_lora_weights:
            print(f"├─ 通用初始化方法: {type(init_lora_weights).__name__}")
            self.reset_lora_parameters(adapter_name, init_lora_weights)
            print(f"└─ 参数重置完成 (方法: {init_lora_weights})")
            
        # 设备同步
        print("\n📡 设备同步:")
        print(f"移动适配器到基础层设备 ({self.get_base_layer().weight.device})")
        self._move_adapter_to_device_of_base_layer(adapter_name)
        print("✅ 设备同步完成")

        # DoRA初始化
        print(f"\n🎯 DoRA配置检测: {'启用' if use_dora else '禁用'}")
        if use_dora:
            print("├─ [DoRA] 开始方向/幅度分解初始化")
            self.dora_init(adapter_name)
            print(f"└─ DoRA配置完成 (方向向量维度: {self.lora_magnitude_vector[adapter_name].weight.shape})")
        self.use_dora[adapter_name] = use_dora
        print(f"DoRA状态更新: {use_dora}")

        # 激活适配器
        print(f"\n⚡ 激活适配器: {self.active_adapters}")
        self.set_adapter(self.active_adapters)
        print("✅ 初始化流程最终完成 ✅")
        print("▛▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▜\n")


    def reset_lora_parameters(self, adapter_name, init_lora_weights):
    # def ok3223():
        print("\n▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜")
        print(f"🔧 开始权重初始化 - 适配器: {adapter_name}")
        print(f"▙ 初始化方法: {init_lora_weights} {'(禁用)' if init_lora_weights is False else ''} ▟")

        if init_lora_weights is False:
            print("\n🛑 检测到初始化禁用标志，跳过初始化流程")
            return

        # 线性层初始化
        if adapter_name in self.lora_A.keys():
            print("\n⚙️ 初始化线性层参数:")
            print(f"LoRA_A形状: {self.lora_A[adapter_name].weight.shape}")
            print(f"LoRA_B形状: {self.lora_B[adapter_name].weight.shape}")

            if init_lora_weights is True:
                print("├─ [默认] 使用Kaiming均匀初始化A，零初始化B")
                nn.init.kaiming_uniform_(
                    self.lora_A[adapter_name].weight, 
                    a=math.sqrt(5)
                )
                print(f"│  ├─ Kaiming均匀初始化参数: a={math.sqrt(5):.4f}")
                
            elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
                std = 1 / self.r[adapter_name]
                print(f"├─ [高斯] 正态分布初始化A (std={std:.4f})")
                nn.init.normal_(
                    self.lora_A[adapter_name].weight,
                    std=std
                )
                
            else:
                error_msg = f"未知初始化方法: {init_lora_weights}"
                print(f"\n❌ 错误: {error_msg}")
                raise ValueError(error_msg)

            # 初始化B矩阵
            print("├─ 零初始化B矩阵权重")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            
            if self.lora_bias[adapter_name]:
                print("└─ 零初始化B矩阵偏置")
                nn.init.zeros_(self.lora_B[adapter_name].bias)
            else:
                print("└─ 未启用B矩阵偏置")

        # 嵌入层初始化
        if adapter_name in self.lora_embedding_A.keys():
            print("\n🔠 初始化嵌入层参数:")
            print(f"嵌入A形状: {self.lora_embedding_A[adapter_name].shape}")
            print(f"嵌入B形状: {self.lora_embedding_B[adapter_name].shape}")

            print("├─ 零初始化嵌入矩阵A")
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            
            print("├─ 正态分布初始化嵌入矩阵B")
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            
            if self.lora_bias[adapter_name]:
                print("└─ 警告: 嵌入层偏置初始化（实验性支持）")
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)
            else:
                print("└─ 未启用嵌入层偏置")

        print("\n✅ 权重初始化完成 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟\n")


    def olora_init(self, adapter_name):
        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight
        bnb_param_type = get_bnb_param_type(orig_weight)
        dtype = orig_weight.dtype

        if bnb_param_type:
            # check without importing bitsandbytes and robust to bnb_4bit_quant_storage=float*
            weight_tensor = dequantize_module_weight(base_layer)
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = orig_weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        if bnb_param_type == "4bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                quant_type=orig_weight.quant_type,
                quant_storage=orig_weight.quant_storage,
                compress_statistics=orig_weight.compress_statistics,
                module=orig_weight.module,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        elif bnb_param_type == "8bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                requires_grad=orig_weight.requires_grad,
                has_fp16_weights=orig_weight.has_fp16_weights,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        else:
            weight_tensor = weight_tensor.to(dtype)
            base_layer.weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = transpose(weight.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight.data = weight

    def loftq_init(self, adapter_name):
        from speft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if self.ephemeral_gpu_offload:
            if lora_A.device.type in ["cuda", "xpu"]:
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type not in ["cuda", "xpu"]:
                    if is_xpu_available():
                        lora_B = lora_B.to("xpu")
                    else:
                        lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
    # def ok3543543():
        print("\n🚀 初始化LoRA适配器层")
        print(f"📌 适配器名称: {adapter_name}")
        print(f"🔧 基础层类型: {type(base_layer).__name__}")

        # 初始化继承链
        print("\n⚙️ 执行父类初始化:")
        super().__init__()
        print("✅ nn.Module 初始化完成")

        print("\n⚙️ 初始化LoraLayer:")
        LoraLayer.__init__(self, base_layer, **kwargs)
        print(f"✅ LoraLayer初始化完成 | 关键参数: { {k:v for k,v in kwargs.items() if not k.startswith('_')} }")

        # 参数配置
        print("\n🔧 配置层参数:")
        self.fan_in_fan_out = fan_in_fan_out
        print(f"   🌀 fan_in_fan_out = {fan_in_fan_out} ({'启用' if fan_in_fan_out else '禁用'})")

        self._active_adapter = adapter_name
        print(f"   📌 激活适配器 = {adapter_name}")

        # 更新层配置
        print("\n🔄 调用update_layer配置:")
        print(f"   🔢 rank(r) = {r}")
        print(f"   α系数 = {lora_alpha} (缩放因子: {lora_alpha/r if r else 'N/A'})")
        print(f"   🎲 Dropout率 = {lora_dropout}")
        print(f"   ⚖️ 权重初始化方式 = {init_lora_weights}")
        print(f"   🌟 RSLoRA = {'启用' if use_rslora else '禁用'}")
        print(f"   🎯 DoRA = {'启用' if use_dora else '禁用'}")
        print(f"   ⚖️ 偏置处理 = {lora_bias or '无'}")

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

        # 特殊卷积标记
        print("\n🔖 设置卷积类型标记:")
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        conv_type = "1D" if is_target_conv_1d_layer else "标准"
        print(f"   🎮 卷积类型 = {conv_type} | 值: {is_target_conv_1d_layer}")

        # 最终设备检查
        print("\n🔍 最终设备状态:")
        if hasattr(self, "lora_A"):
            device = self.lora_A[adapter_name].weight.device
            print(f"   🔗 LoRA参数设备: {device}")
        else:
            print("⚠️ 未检测到LoRA参数矩阵")

        print("\n✅ LoRA层初始化完成")
        print("="*60)


    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        if lora_bias:
            # lora_bias=True is not supported (yet) for embedding layers, as they use nn.Parameter
            raise ValueError(f"lora_bias={lora_bias} is not supported for {self.__class__.__name__}.")

        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraEmbeddingLayer(fan_in_fan_out=True)
        lora_embedding_A = self.lora_embedding_A[adapter_name]
        lora_embedding_B = self.lora_embedding_B[adapter_name]
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_embedding_A, lora_B=lora_embedding_B, scaling=scaling
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]

                if not self.use_dora[active_adapter]:
                    after_A = self._embed(x, embedding_A)
                    result = result + (after_A @ embedding_B) * scaling
                else:
                    mag_norm_scale, dora_result = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=embedding_A,
                        lora_B=embedding_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        embed_fn=self._embed,
                    )
                    result = mag_norm_scale * result + dora_result
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class _ConvNd(nn.Module, LoraLayer):
    # Lora implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.lora_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = conv_layer(r, self.out_features, out_kernel, out_stride, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def _get_dora_factor_view(self):
        return (-1,) + (1,) * (self._kernel_dim - 1)

    def dora_init(self, adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer_class = self._get_dora_layer_class()
        dora_layer = dora_layer_class(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _get_dora_layer_class(self) -> type[_DoraConvNdLayer]:
        # Subclasses should override this method to return the appropriate DoraLayer class
        raise NotImplementedError

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(*self._get_dora_factor_view()) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(*self._get_dora_factor_view()) * (
                            base_layer.weight.data + delta_weight
                        )
                        base_layer.weight.data = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(*self._get_dora_factor_view()) - delta_weight
                    weight.data = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            output_tensor = (
                self.conv_fn(
                    weight_A.transpose(0, 1),
                    weight_B,
                ).transpose(0, 1)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(_ConvNd):
    # Lora implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d

    def _get_dora_layer_class(self):
        return DoraConv2dLayer


class Conv3d(_ConvNd):
    # Lora implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d

    def _get_dora_layer_class(self):
        return DoraConv3dLayer


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
# def ok2432():
    print("\n🚀 开始创建适配器模块")
    print(f"📌 目标层类型: {type(target).__name__}")
    print(f"🔧 适配器名称: {adapter_name}")
    print(f"⚙️ 初始参数: { {k:v for k,v in kwargs.items() if not isinstance(v, dict)} }")

    new_module = None
    
    # 获取基础层
    print("\n🔍 解析基础层类型:")
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
        print(f"   🎯 基础层来自TunerLayer: {type(target_base_layer).__name__}")
    else:
        target_base_layer = target
        print(f"   🎯 直接使用目标层: {type(target_base_layer).__name__}")

    # Embedding处理
    if isinstance(target_base_layer, torch.nn.Embedding):
        print("\n📦 处理Embedding层:")
        embedding_kwargs = kwargs.copy()
        print(f"   📥 复制原始参数（排除fan_in_fan_out）")
        embedding_kwargs.pop("fan_in_fan_out", None)
        
        print(f"   🔄 合并LoFTQ配置: {lora_config.loftq_config}")
        embedding_kwargs.update(lora_config.loftq_config)
        
        print(f"🎯 创建Embedding适配器")
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
        print(f"✅ 模块创建成功 | 类型: {type(new_module).__name__}")

    # Conv2D处理
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        print("\n🎮 处理Conv2D层:")
        print(f"   🔄 合并LoFTQ配置: {lora_config.loftq_config}")
        kwargs.update(lora_config.loftq_config)
        
        print(f"🎯 创建Conv2D适配器")
        new_module = Conv2d(target, adapter_name, **kwargs)
        print(f"✅ 模块创建成功 | 参数: kernel_size={target_base_layer.kernel_size}")

    # Conv3D处理 
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        print("\n🎮 处理Conv3D层:")
        print(f"   🔄 合并LoFTQ配置: {lora_config.loftq_config}")
        kwargs.update(lora_config.loftq_config)
        
        print(f"🎯 创建Conv3D适配器") 
        new_module = Conv3d(target, adapter_name, **kwargs)
        print(f"✅ 模块创建成功 | 参数: kernel_size={target_base_layer.kernel_size}")

    # Linear处理
    elif isinstance(target_base_layer, torch.nn.Linear):
        print("\n📏 处理Linear层:")
        if kwargs["fan_in_fan_out"]:
            print("⚠️ 警告: 检测到fan_in_fan_out=True与Linear层冲突")
            print("   🔧 自动修正为fan_in_fan_out=False")
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            
        print(f"   🔄 合并LoFTQ配置: {lora_config.loftq_config}")
        kwargs.update(lora_config.loftq_config)
        
        print(f"🎯 创建Linear适配器")
        new_module = Linear(target, adapter_name, **kwargs)
        print(f"✅ 模块创建成功 | 特征数: in={target_base_layer.in_features}, out={target_base_layer.out_features}")

    # Conv1D处理
    elif isinstance(target_base_layer, Conv1D): 
        print("\n📏 处理Conv1D层:")
        if not kwargs["fan_in_fan_out"]:
            print("⚠️ 警告: 检测到fan_in_fan_out=False与Conv1D层冲突")
            print("   🔧 自动修正为fan_in_fan_out=True")
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            
        print(f"   🔄 合并LoFTQ配置: {lora_config.loftq_config}")
        kwargs.update(lora_config.loftq_config)
        
        print(f"🎯 创建Conv1D适配器")
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)
        print(f"✅ 模块创建成功 | 权重形状: {target_base_layer.weight.shape}")


    return new_module
