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
"""
PyTorch utilities: Utilities related to PyTorch
"""

from typing import List, Optional, Tuple, Union

import diffusers.utils.logging as logging
from diffusers.utils.import_utils import is_torch_available, is_torch_version


if is_torch_available():
    import torch
    from torch.fft import fftn, fftshift, ifftn, ifftshift

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
# def ok23432():
    print("\n[Latent Generation] 开始潜在变量生成")
    
    # 初始化设备与布局
    print("\n[阶段1] 设备配置")
    print(f"⚙️ 输入设备: {device or '未指定，使用默认CPU'}")
    rand_device = device
    batch_size = shape[0]
    layout = layout or torch.strided
    print(f"✅ 最终配置: device={rand_device}, layout={layout}, batch_size={batch_size}")

    # 生成器设备验证
    if generator is not None:
        print("\n[阶段2] 生成器设备检查")
        if isinstance(generator, list):
            gen_device_type = generator[0].device.type
            print(f"🔍 检测到生成器列表 (长度: {len(generator)}), 首元素设备: {gen_device_type}")
        else:
            gen_device_type = generator.device.type
            print(f"🔍 单一生成器设备类型: {gen_device_type}")

        # 设备兼容性处理
        if gen_device_type != rand_device.type:
            if gen_device_type == "cpu":
                print("⚠️ 生成器在CPU但目标设备为GPU，潜在性能影响")
                print("   原因: CPU生成器需额外数据传输至GPU")
                rand_device = "cpu"
            elif gen_device_type == "cuda":
                raise ValueError("❌ 生成器在CUDA但目标设备非GPU，设备冲突")
            print(f"🔄 调整生成设备为: {rand_device}")
    else:
        print("\n⚙️ 未提供生成器，使用默认随机种子")

    # 生成器列表处理
    print("\n[阶段3] 生成器类型处理")
    if isinstance(generator, list):
        if len(generator) == 1:
            print("⚡ 将单元素生成器列表转换为独立生成器")
            generator = generator[0]
            print(f"   → 新生成器类型: {type(generator).__name__}")
    
    # 潜在变量生成
    print("\n[阶段4] 噪声生成")
    if isinstance(generator, list):
        print(f"🔢 按批次独立生成 (数量: {batch_size})")
        sub_shape = (1,) + shape[1:]
        latents = []
        for i in range(batch_size):
            print(f"   → 生成批次 {i+1}/{batch_size}")
            print(f"      子形状: {sub_shape}")
            print(f"      生成器: {generator[i].device if generator[i] else '默认'}")
            latent = torch.randn(sub_shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            latents.append(latent)
        latents = torch.cat(latents, dim=0).to(device)
        print(f"✅ 合并后形状: {latents.shape}")
    else:
        print(f"⚡ 批量生成潜在变量 (形状: {shape})")
        print(f"   生成器设备: {generator.device if generator else '默认'}")
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
    
    # 最终设备验证
    print("\n[阶段5] 设备迁移验证")
    print(f"🔄 潜在变量当前设备: {latents.device}")
    if latents.device != device:
        print(f"⚠️ 设备不匹配: 预期 {device}, 实际 {latents.device}")
        latents = latents.to(device)
        print(f"✅ 已迁移至目标设备: {latents.device}")
    
    # 统计信息
    print("\n📊 生成结果统计:")
    print(f"   → 形状: {latents.shape}")
    print(f"   → 数据类型: {latents.dtype}")
    print(f"   → 均值: {latents.mean().item():.4f}")
    print(f"   → 标准差: {latents.std().item():.4f}")
    print(f"   → 值域: [{latents.min().item():.4f}, {latents.max().item():.4f}]")

    print("\n[Latent Generation] 生成完成 ✅\n")
    return latents



def is_compiled_module(module) -> bool:
    """Check whether the module was compiled with torch.compile()"""
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int) -> "torch.Tensor":
    """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

    This version of the method comes from here:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """
    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)
    # fftn does not support bfloat16
    elif x.dtype == torch.bfloat16:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fftn(x, dim=(-2, -1))
    x_freq = fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = ifftshift(x_freq, dim=(-2, -1))
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)


def apply_freeu(
    resolution_idx: int, hidden_states: "torch.Tensor", res_hidden_states: "torch.Tensor", **freeu_kwargs
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Applies the FreeU mechanism as introduced in https:
    //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

    Args:
        resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
        hidden_states (`torch.Tensor`): Inputs to the underlying block.
        res_hidden_states (`torch.Tensor`): Features from the skip block corresponding to the underlying block.
        s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
        s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
        b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
        b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
    """
    if resolution_idx == 0:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b1"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s1"])
    if resolution_idx == 1:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b2"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s2"])

    return hidden_states, res_hidden_states
