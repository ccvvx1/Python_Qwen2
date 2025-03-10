# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.normalization import RMSNorm
from diffusers.models.upsampling import upfirdn2d_native


class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        return self.conv(inputs)


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
    # def ok32432():
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        print("\n=== 下采样模块初始化 ===")
        print(f"初始参数验证:")
        print(f"  ├─ 输入通道: {channels} (类型: {type(channels)})")
        print(f"  ├─ 输出通道: {out_channels or '未指定，使用输入通道'} (最终值: {out_channels})")
        print(f"  ├─ 下采样方式: {'可学习卷积' if use_conv else '平均池化'}")
        print(f"  └─ 填充配置: {padding} (类型检查: {type(padding).__name__})")

        # 参数合法性检查
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"通道数必须为正整数，当前值: {channels}")

        # 标准化层初始化跟踪
        print("\n[标准化层配置]")
        if norm_type == "ln_norm":
            print(f"  ├─ 类型: LayerNorm (torch原生实现)")
            print(f"  │   ├─ eps: {eps:.1e}")
            print(f"  │   └─ 可学习参数: {'启用' if elementwise_affine else '禁用'}")
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            print(f"  ├─ 类型: RMSNorm (自定义实现)")
            print(f"  │   ├─ eps: {eps:.1e}")
            print(f"  │   └─ 可学习参数: {'启用' if elementwise_affine else '禁用'}")
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            print("  ├─ 未启用标准化层")
            self.norm = None
        else:
            error_msg = (
                f"未知的标准化类型 '{norm_type}'\n"
                f"可用选项: ['ln_norm'(LayerNorm), 'rms_norm'(RMSNorm), None]\n"
                f"常见错误原因:\n"
                f"  1. 配置文件中的拼写错误\n"
                f"  2. 版本不兼容导致参数变更"
            )
            print(f"[初始化失败] {error_msg}")
            raise ValueError(error_msg)

        # 下采样层构建跟踪
        print("\n[下采样核心层构建]")
        if use_conv:
            print(f"  ├─ 卷积层配置:")
            print(f"      ├─ 输入通道: {self.channels}")
            print(f"      ├─ 输出通道: {self.out_channels}")
            print(f"      ├─ 核尺寸: {kernel_size} (类型: {type(kernel_size).__name__})")
            print(f"      ├─ 步长: {stride} (固定值)")
            print(f"      └─ 偏置: {'启用' if bias else '禁用'}")
            
            # 卷积参数有效性检查
            if kernel_size % 2 == 0:
                print(f"  ⚠️ 警告: 偶数尺寸卷积核({kernel_size})可能导致不对称填充问题")
            
            conv = nn.Conv2d(
                self.channels, self.out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=bias
            )
        else:
            print(f"  ├─ 平均池化配置:")
            print(f"      ├─ 池化窗口: {stride}x{stride}")
            print(f"      └─ 通道验证: 输入/输出均为 {self.channels} 通道")
            try:
                assert self.channels == self.out_channels
            except AssertionError as e:
                print(f"  [!] 通道不匹配错误: 输入({self.channels}) ≠ 输出({self.out_channels})")
                print(f"      解决方案: 当使用池化时，out_channels必须等于输入通道数或保持None")
                raise
            
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

    # def ok23432():
        print("\n=== 卷积层命名兼容性处理 ===")
        print(f"传入的模块名称: '{name}' (类型: {type(name)})")
        
        # 显示当前卷积层配置
        conv_info = f"Conv2d(in={conv.in_channels}, out={conv.out_channels}, "
        if isinstance(conv, nn.Conv2d):
            conv_info += f"kernel={conv.kernel_size}, stride={conv.stride})"
        else:
            conv_info += f"pool_type=AvgPool2d)"
        print(f"待分配卷积层参数: {conv_info}")

        # 命名兼容性处理分支
        if name == "conv":
            print("\n!! 历史命名兼容模式 !!")
            print("  检测到 name='conv' 的旧版命名方式")
            print("  同时设置两个属性: Conv2d_0 和 conv")
            print("  [!] 警告: 这是为了兼容旧权重字典的临时方案")
            print("      建议: 完成权重迁移后应统一使用单一属性名")
            
            self.Conv2d_0 = conv
            self.conv = conv
            print(f"  已绑定属性: {hasattr(self, 'Conv2d_0')} | {hasattr(self, 'conv')}")

        elif name == "Conv2d_0":
            print("\n!! 过渡命名模式 !!")
            print("  检测到 name='Conv2d_0' 的中间命名规范")
            print("  现代实现应直接使用 'conv' 作为属性名")
            print("  正在设置单一属性: conv")
            
            self.conv = conv
            print(f"  属性存在检查: conv => {hasattr(self, 'conv')}")

        else:
            print("\n!! 非常规命名模式 !!")
            print(f"  使用非标准名称: '{name}'")
            print("  可能影响:")
            print("    1. 预训练权重加载兼容性")
            print("    2. 模型可视化工具识别")
            print("    3. 序列化/反序列化一致性")
            
            self.conv = conv
            print(f"  最终绑定属性: ['conv'] (实际使用名称: '{name}')")




    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class FirDownsample2D(nn.Module):
    """A 2D FIR downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        fir_kernel: Tuple[int, int, int, int] = (1, 3, 3, 1),
    ):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_2d(
        self,
        hidden_states: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        kernel: Optional[torch.Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ) -> torch.Tensor:
        """Fused `Conv2d()` followed by `downsample_2d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight (`torch.Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`torch.Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to average pooling.
            factor (`int`, *optional*, default to `2`):
                Integer downsampling factor.
            gain (`float`, *optional*, default to `1.0`):
                Scaling factor for signal magnitude.

        Returns:
            output (`torch.Tensor`):
                Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and same
                datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)

        kernel = kernel * gain

        if self.use_conv:
            _, _, convH, convW = weight.shape
            pad_value = (kernel.shape[0] - factor) + (convW - 1)
            stride_value = [factor, factor]
            upfirdn_input = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            output = F.conv2d(upfirdn_input, weight, stride=stride_value, padding=0)
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )

        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            downsample_input = self._downsample_2d(hidden_states, weight=self.Conv2d_0.weight, kernel=self.fir_kernel)
            hidden_states = downsample_input + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            hidden_states = self._downsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

        return hidden_states


# downsample/upsample layer used in k-upscaler, might be able to use FirDownsample2D/DirUpsample2D instead
class KDownsample2D(nn.Module):
    r"""A 2D K-downsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer("kernel", kernel_1d.T @ kernel_1d, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.pad(inputs, (self.pad,) * 4, self.pad_mode)
        weight = inputs.new_zeros(
            [
                inputs.shape[1],
                inputs.shape[1],
                self.kernel.shape[0],
                self.kernel.shape[1],
            ]
        )
        indices = torch.arange(inputs.shape[1], device=inputs.device)
        kernel = self.kernel.to(weight)[None, :].expand(inputs.shape[1], -1, -1)
        weight[indices, indices] = kernel
        return F.conv2d(inputs, weight, stride=2)


class CogVideoXDownsample3D(nn.Module):
    # Todo: Wait for paper relase.
    r"""
    A 3D Downsampling layer using in [CogVideoX]() by Tsinghua University & ZhipuAI

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `2`):
            Stride of the convolution.
        padding (`int`, defaults to `0`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape

            # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.conv(x)
        # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x


def downsample_2d(
    hidden_states: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    r"""Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states (`torch.Tensor`)
            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel (`torch.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to average pooling.
        factor (`int`, *optional*, default to `2`):
            Integer downsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude.

    Returns:
        output (`torch.Tensor`):
            Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output
