# Copyright 2024 The HuggingFace Team. All rights reserved.
# `TemporalConvLayer` Copyright 2024 Alibaba DAMO-VILAB, The ModelScope Team and The HuggingFace Team. All rights reserved.
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

from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from downsampling import (  # noqa
    Downsample1D,
    Downsample2D,
    FirDownsample2D,
    KDownsample2D,
    downsample_2d,
)
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.upsampling import (  # noqa
    FirUpsample2D,
    KUpsample2D,
    Upsample1D,
    Upsample2D,
    upfirdn2d_native,
    upsample_2d,
)


class ResnetBlockCondNorm2D(nn.Module):
    r"""
    A Resnet block that use normalization layer that incorporate conditioning information.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"ada_group"` ):
            The normalization layer for time embedding `temb`. Currently only support "ada_group" or "spatial".
        kernel (`torch.Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "ada_group",  # ada_group, spatial
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":  # ada_group
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.time_embedding_norm == "ada_group":  # ada_group
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":  # spatial
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

        self.dropout = torch.nn.Dropout(dropout)

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states, temb)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)

        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, temb)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" for a
            stronger conditioning with scale and shift.
        kernel (`torch.Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        print("*"*60)
        super().__init__()

        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
            )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        print(f"\n[初始化] groups_out 初始值: {groups_out}（若为None将继承groups值）")
        if groups_out is None:
            groups_out = groups
            print(f"※ groups_out 自动设置为与groups相同: {groups_out}")

        print(f"\n[GroupNorm1] 创建分组归一化层｜分组数={groups} 输入通道={in_channels} 精度={eps}")
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        print(f"✓ GroupNorm1已生成｜实际分组数={self.norm1.num_groups} 处理通道={self.norm1.num_channels}")

        print(f"\n[卷积层1] 构建3x3卷积｜输入通道={in_channels} 输出通道={out_channels}")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        print(f"✓ Conv1权重形状={tuple(self.conv1.weight.shape)} 偏置={self.conv1.bias is not None}")

        if temb_channels is not None:
            print(f"\n[时间嵌入] 检测到时间嵌入维度: {temb_channels}")
            print(f"模式选择: {self.time_embedding_norm}")
            if self.time_embedding_norm == "default":
                print(f"创建线性投影层: {temb_channels} → {out_channels}")
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                print(f"创建缩放偏移投影: {temb_channels} → {2 * out_channels}")
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"不支持的归一化模式: {self.time_embedding_norm} ")
            print(f"※ 时间嵌入层已创建: {type(self.time_emb_proj)}")
        else:
            print("\n[时间嵌入] 未提供时间嵌入参数，跳过投影层")
            self.time_emb_proj = None

        print(f"\n[GroupNorm2] 创建次级归一化｜分组数={groups_out} 通道数={out_channels}")
        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        print(f"✓ GroupNorm2实际参数｜分组数={self.norm2.num_groups} 处理通道={self.norm2.num_channels}")

        print(f"\n[Dropout] 初始化随机丢弃层｜丢弃概率={dropout}")
        self.dropout = torch.nn.Dropout(dropout)

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        print(f"\n[卷积层2] 最终3x3卷积｜输入={out_channels} 输出={conv_2d_out_channels}")
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
        print(f"✓ Conv2权重形状={tuple(self.conv2.weight.shape)} 偏置={self.conv2.bias is not None}")

        print(f"\n[激活函数] 正在加载: {non_linearity}")
        self.nonlinearity = get_activation(non_linearity)
        print(f"※ 激活函数实例: {type(self.nonlinearity)}")

        print(f"\n[采样层] UP标志状态: {self.up}")
        self.upsample = self.downsample = None
        if self.up:
            print(f"\n[上采样] 检测到up标志为True，开始初始化｜kernel类型={kernel}")
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                print(f"※ 使用FIR滤波器上采样｜内核形状{fir_kernel}")
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
                print("✓ 已创建lambda函数upsample_2d")
            elif kernel == "sde_vp":
                print("※ 使用SDE_VP模式｜最近邻插值2倍缩放")
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
                print(f"采样函数类型：{type(self.upsample)}")
            else:
                print(f"使用默认上采样器｜输入通道={in_channels}")
                self.upsample = Upsample2D(in_channels, use_conv=False)
                print(f"上采样层结构：{self.upsample.__class__.__name__}")

        elif self.down:
            print(f"\n[下采样] 检测到down标志为True｜kernel类型={kernel}")
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                print(f"※ 使用FIR滤波器下采样｜内核形状{fir_kernel}")
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
                print("✓ 已创建lambda函数downsample_2d")
            elif kernel == "sde_vp":
                print("※ 使用SDE_VP模式｜平均池化2x2")
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
                print(f"池化函数类型：{type(self.downsample)}")
            else:
                print(f"使用默认下采样器｜输入通道={in_channels} 填充=1")
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")
                print(f"下采样层结构：{self.downsample.__class__.__name__}")

        print(f"\n[捷径连接] 初始化检查｜输入通道={in_channels} 输出通道={conv_2d_out_channels}")
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut
        print(f"※ 自动判断是否需要捷径连接：{'需要' if self.use_in_shortcut else '不需要'}")

        self.conv_shortcut = None
        if self.use_in_shortcut:
            print(f"\n[创建捷径卷积] 1x1卷积｜in={in_channels} out={conv_2d_out_channels} 偏置={conv_shortcut_bias}")
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )
            print(f"✓ 捷径卷积创建成功｜权重形状={tuple(self.conv_shortcut.weight.shape)}")
        else:
            print("\n[捷径连接] 通道数匹配，跳过1x1卷积")


    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        print("*" * 60)
        # 弃用警告处理
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            print("\n⚠️ 检测到过时参数使用".center(50, "-"))
            deprecation_message = "`scale`参数已弃用，请通过cross_attention_kwargs传递"
            print(f"警告版本：v1.0.0｜详细信息：{deprecation_message}")
            deprecate("scale", "1.0.0", deprecation_message)

        # 初始状态记录
        print(f"\n[前向传播] 输入形状: {input_tensor.shape}｜初始隐藏状态相同")
        hidden_states = input_tensor

        # 归一化与激活
        print(f"\n[预处理] 应用GroupNorm1 + {self.nonlinearity.__class__.__name__}")
        hidden_states = self.norm1(hidden_states)
        print(f"归一化后均值={hidden_states.mean().item():.4f} 方差={hidden_states.var().item():.4f}")
        hidden_states = self.nonlinearity(hidden_states)
        print(f"激活后极值：max={hidden_states.max().item():.2f} min={hidden_states.min().item():.2f}")

        # 采样处理
        if self.upsample is not None:
            print(f"\n[上采样] 检测到上采样器｜批次大小={hidden_states.shape[0]}")
            if hidden_states.shape[0] >= 64:
                print("※ 大批次处理(>=64)｜启用contiguous()")
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            print(f"上采样前形状：input_tensor={input_tensor.shape} hidden_states={hidden_states.shape}")
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
            print(f"✓ 上采样后形状：input_tensor={input_tensor.shape} hidden_states={hidden_states.shape}")

        elif self.downsample is not None:
            print(f"\n[下采样] 检测到下采样器")
            print(f"采样前形状：input_tensor={input_tensor.shape} hidden_states={hidden_states.shape}")
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)
            print(f"✓ 下采样后形状：input_tensor={input_tensor.shape} hidden_states={hidden_states.shape}")

        # 卷积层1
        print(f"\n[卷积1] 应用{self.conv1}")
        hidden_states = self.conv1(hidden_states)
        print(f"卷积输出形状：{hidden_states.shape}｜通道数={hidden_states.size(1)}")

        # 时间嵌入处理
        if self.time_emb_proj is not None:
            print(f"\n[时间嵌入] 投影层存在｜skip_time_act={self.skip_time_act}")
            if not self.skip_time_act:
                print("※ 应用非线性激活到temb")
                temb = self.nonlinearity(temb)
                print(f"激活后temb统计：max={temb.max().item():.2f} min={temb.min().item():.2f}")
            temb = self.time_emb_proj(temb)[:, :, None, None]
            print(f"投影后形状：{temb.shape}")

        # 时间归一化分支
        print(f"\n[时间归一化] 模式={self.time_embedding_norm}")
        if self.time_embedding_norm == "default":
            print("→ default模式处理")
            if temb is not None:
                print(f"添加时间嵌入｜hidden_states形状：{hidden_states.shape} temb形状：{temb.shape}")
                hidden_states = hidden_states + temb
                print(f"融合后极值：max={hidden_states.max().item():.2f} min={hidden_states.min().item():.2f}")
            hidden_states = self.norm2(hidden_states)
            print(f"GroupNorm2输出均值={hidden_states.mean().item():.4f}")

        elif self.time_embedding_norm == "scale_shift":
            print("→ scale_shift模式处理")
            if temb is None:
                error_msg = f"scale_shift模式需要temb输入，当前temb=None"
                print(f"❌ 严重错误：{error_msg}")
                raise ValueError(error_msg)
            
            print(f"分割scale/shift｜输入形状：{temb.shape}")
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            print(f"scale形状：{time_scale.shape} shift形状：{time_shift.shape}")
            
            hidden_states = self.norm2(hidden_states)
            print(f"归一化后均值={hidden_states.mean().item():.4f}")
            
            hidden_states = hidden_states * (1 + time_scale) + time_shift
            print(f"缩放偏移后极值：max={hidden_states.max().item():.2f} min={hidden_states.min().item():.2f}")

        else:
            print("→ 直接应用GroupNorm2")
            hidden_states = self.norm2(hidden_states)
            print(f"归一化后统计：均值={hidden_states.mean().item():.4f} 方差={hidden_states.var().item():.4f}")

        # 非线性激活
        print(f"\n[激活] 应用{self.nonlinearity.__class__.__name__}")
        hidden_states = self.nonlinearity(hidden_states)
        print(f"激活后统计: max={hidden_states.max().item():.2f} min={hidden_states.min().item():.2f}")
        print(f"均值={hidden_states.mean().item():.4f} 标准差={hidden_states.std().item():.4f}")

        # Dropout处理
        print(f"\n[Dropout] 模式={'训练' if self.training else '评估'}｜丢弃率={self.dropout.p}")
        hidden_states = self.dropout(hidden_states)
        if self.training:
            zero_ratio = (hidden_states == 0).float().mean().item()
            print(f"丢弃后零值占比: {zero_ratio*100:.1f}%")
        else:
            print("※ 评估模式下Dropout未激活")

        # 最终卷积层
        print(f"\n[卷积2] 应用{self.conv2}")
        hidden_states = self.conv2(hidden_states)
        print(f"输出形状: {hidden_states.shape}｜输出通道={hidden_states.size(1)}")
        print(f"卷积核均值: {self.conv2.weight.mean().item():.4f}")

        # 捷径连接处理
        if self.conv_shortcut is not None:
            print(f"\n[捷径卷积] 应用1x1卷积调整输入通道")
            print(f"调整前input_tensor形状: {input_tensor.shape}")
            input_tensor = self.conv_shortcut(input_tensor)
            print(f"✓ 调整后input_tensor形状: {input_tensor.shape}")
            print(f"捷径卷积权重范数: {torch.norm(self.conv_shortcut.weight).item():.2f}")
        else:
            print("\n[捷径连接] 无通道调整，直接使用原始input_tensor")

        # 残差连接
        print(f"\n[残差加和] 合并主路径与捷径")
        print(f"主路径形状: {hidden_states.shape}｜捷径形状: {input_tensor.shape}")
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        print(f"合并后极值: max={output_tensor.max().item():.2f} min={output_tensor.min().item():.2f}")
        print(f"缩放因子: {self.output_scale_factor} → 最终均值={output_tensor.mean().item():.4f}")
        print(f"输出张量形状: {output_tensor.shape}")


        return output_tensor


# unet_rl.py
def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_groups: int = 8,
        activation: str = "mish",
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = get_activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        intermediate_repr = self.conv1d(inputs)
        intermediate_repr = rearrange_dims(intermediate_repr)
        intermediate_repr = self.group_norm(intermediate_repr)
        intermediate_repr = rearrange_dims(intermediate_repr)
        output = self.mish(intermediate_repr)
        return output


# unet_rl.py
class ResidualTemporalBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        embed_dim (`int`): Embedding dimension.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        activation (`str`, defaults `mish`): It is possible to choose the right activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: str = "mish",
    ):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        self.time_emb_act = get_activation(activation)
        self.time_emb = nn.Linear(embed_dim, out_channels)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(inputs) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016

    Parameters:
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states


class TemporalResnetBlock(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.nonlinearity = get_activation("silu")

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


# VideoResBlock
class SpatioTemporalResBlock(nn.Module):
    r"""
    A SpatioTemporal Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the spatial resenet.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states


class AlphaBlender(nn.Module):
    r"""
    A module to blend spatial and temporal features.

    Parameters:
        alpha (`float`): The initial value of the blending factor.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix  # For TemporalVAE

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor, ndims: int) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor

        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)

        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError("Please provide image_only_indicator to use learned_with_images merge strategy")

            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                torch.sigmoid(self.mix_factor)[..., None],
            )

            # (batch, channel, frames, height, width)
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError

        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        alpha = alpha.to(x_spatial.dtype)

        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x
