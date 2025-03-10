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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, unscale_lora_layers
from peft_utils import scale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from embeddings import (
    GaussianFourierProjection,
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class UNet2DConditionModel(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin
):
    r"""
    A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            Block type for middle of UNet, it can be one of `UNetMidBlock2DCrossAttn`, `UNetMidBlock2D`, or
            `UNetMidBlock2DSimpleCrossAttn`. If `None`, the mid block layer is skipped.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unets.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unets.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unets.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        reverse_transformer_layers_per_block : (`Tuple[Tuple]`, *optional*, defaults to None):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`], in the upsampling
            blocks of the U-Net. Only relevant if `transformer_layers_per_block` is of type `Tuple[Tuple]` and for
            [`~models.unets.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unets.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unets.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        num_attention_heads (`int`, *optional*):
            The number of attention heads. If not defined, defaults to `attention_head_dim`
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        addition_time_embed_dim: (`int`, *optional*, defaults to `None`):
            Dimension for the timestep embeddings.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, defaults to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, defaults to `None`):
            An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, defaults to `None`):
            Optional activation function to use only once on the time embeddings before they are passed to the rest of
            the UNet. Choose from `silu`, `mish`, `gelu`, and `swish`.
        timestep_post_act (`str`, *optional*, defaults to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, defaults to `None`):
            The dimension of `cond_proj` layer in the timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            `class_embed_type="projection"`. Required when `class_embed_type="projection"`.
        class_embeddings_concat (`bool`, *optional*, defaults to `False`): Whether to concatenate the time
            embeddings with the class embeddings.
        mid_block_only_cross_attention (`bool`, *optional*, defaults to `None`):
            Whether to use cross attention with the mid block when using the `UNetMidBlock2DSimpleCrossAttn`. If
            `only_cross_attention` is given as a single boolean and `mid_block_only_cross_attention` is `None`, the
            `only_cross_attention` value is used as the value for `mid_block_only_cross_attention`. Default to `False`
            otherwise.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
    # def ok323232():
        super().__init__()
        print("\n[UNet Initialization] 开始模型初始化")
        
        # 参数有效性验证
        print("\n[阶段1] 参数校验")
        self.sample_size = sample_size
        print(f"✅ 样本尺寸: {sample_size}")
        
        if num_attention_heads is not None:
            error_msg = (
                "⚠️ 当前版本不支持显式设置num_attention_heads参数\n"
                "  原因: 参数命名冲突问题 (详见 https://github.com/huggingface/diffusers/issues/2011)\n"
                "  解决方案: 请使用attention_head_dim参数或等待diffusers v0.19版本"
            )
            print(error_msg)
            raise ValueError(error_msg)
        
        num_attention_heads = num_attention_heads or attention_head_dim
        print(f"⚙️ 最终注意力头数: {num_attention_heads}")

        # 配置检查
        print("\n[阶段2] 配置完整性检查")
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
        )
        print("✅ 所有配置参数通过校验")

        # 输入处理
        print("\n[阶段3] 输入层构建")
        conv_in_padding = (conv_in_kernel - 1) // 2
        print(f"🔧 卷积层参数:")
        print(f"   → 输入通道: {in_channels}")
        print(f"   → 输出通道: {block_out_channels[0]}")
        print(f"   → 核大小: {conv_in_kernel}x{conv_in_kernel}")
        print(f"   → 自动填充计算: ({conv_in_kernel}-1)//2 = {conv_in_padding}")
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        print(f"✅ 输入卷积层构建完成: {self.conv_in}")

        # 时间嵌入
        print("\n[阶段4] 时间嵌入配置")
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )
        print(f"⚙️ 时间投影参数:")
        print(f"   → 输入维度: {timestep_input_dim}")
        print(f"   → 嵌入维度: {time_embed_dim}")
        print(f"   → 频率偏移: {freq_shift or '无'}")
        print(f"   → 波形转换: {'sin→cos' if flip_sin_to_cos else '保持原始'}")

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        print(f"✅ 时间嵌入层构建完成:")
        print(f"   → 激活函数: {act_fn}")
        print(f"   → 后处理函数: {timestep_post_act or '无'}")
        print(f"   → 条件投影维度: {time_cond_proj_dim or '未启用'}")

    # def okew432432432():
        print("\n[UNet Configuration] 开始高级配置")
        
        # 编码器隐藏层投影
        print("\n[阶段1] 编码器隐藏投影设置")
        print(f"🔧 参数列表:")
        print(f"   → encoder_hid_dim_type: {encoder_hid_dim_type}")
        print(f"   → cross_attention_dim: {cross_attention_dim}")
        print(f"   → encoder_hid_dim: {encoder_hid_dim}")
        self._set_encoder_hid_proj(
            encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )
        print("✅ 编码器投影配置完成")

        # 类别嵌入配置
        print("\n[阶段2] 类别嵌入配置")
        print(f"⚙️ 嵌入类型: {class_embed_type}")
        print(f"   → 类别数量: {num_class_embeds or '无'}")
        print(f"   → 投影输入维度: {projection_class_embeddings_input_dim or '未启用'}")
        self._set_class_embedding(
            class_embed_type,
            act_fn=act_fn,
            num_class_embeds=num_class_embeds,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )
        # print(f"✅ 类别嵌入维度: {self.class_embedding.out_features if hasattr(self, 'class_embedding') else '未启用'}")

        # 附加嵌入配置
        print("\n[阶段3] 附加嵌入设置")
        print(f"🔧 嵌入类型: {addition_embed_type}")
        print(f"   → 多头数量: {addition_embed_type_num_heads}")
        print(f"   → 时间嵌入维度: {addition_time_embed_dim}")
        self._set_add_embedding(
            addition_embed_type,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )
        print(f"✅ 附加嵌入参数: {self.add_embedding.config if hasattr(self, 'add_embedding') else '未启用'}")

        # 时间嵌入激活函数
        print("\n[阶段4] 激活函数配置")
        if time_embedding_act_fn is None:
            print("⚙️ 未设置时间嵌入激活函数")
            self.time_embed_act = None
        else:
            print(f"🔄 初始化激活函数: {time_embedding_act_fn}")
            self.time_embed_act = get_activation(time_embedding_act_fn)
            print(f"✅ 激活函数对象: {self.time_embed_act}")

        # 初始化网络块
        print("\n[阶段5] 网络架构构建")
        print(f"🛠️ 初始化下采样块 ({len(down_block_types)}个)")
        self.down_blocks = nn.ModuleList([])
        print(f"🛠️ 初始化上采样块 ({len(up_block_types)}个)")
        self.up_blocks = nn.ModuleList([])
        print(f"📐 中间块交叉注意力配置: {mid_block_only_cross_attention}")

        # 交叉注意力参数处理
        print("\n[阶段6] 注意力机制参数统一")
        if isinstance(only_cross_attention, bool):
            print(f"🔧 统一交叉注意力参数 (原值: {only_cross_attention})")
            if mid_block_only_cross_attention is None:
                print(f"   → 自动设置中间块参数为: {only_cross_attention}")
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)
            print(f"✅ 参数扩展结果: {only_cross_attention}")
        else:
            print(f"⚙️ 使用自定义交叉注意力配置: {only_cross_attention}")

    # def ok323232():
        print("\n[UNet Parameter Uniformization] 开始参数统一化处理")
        
        # 中间块交叉注意力默认处理
        print("\n[阶段1] 中间块配置")
        if mid_block_only_cross_attention is None:
            print("⚙️ 未指定中间块交叉注意力，设置为默认值False")
            mid_block_only_cross_attention = False
        print(f"✅ 最终中间块交叉注意力: {mid_block_only_cross_attention}")

        # 参数扩展处理
        print("\n[阶段2] 参数维度扩展")
        param_config = [
            ("注意力头数(num_attention_heads)", num_attention_heads, len(down_block_types)),
            ("头维度(attention_head_dim)", attention_head_dim, len(down_block_types)),
            ("交叉注意力维度(cross_attention_dim)", cross_attention_dim, len(down_block_types)),
            ("每块层数(layers_per_block)", layers_per_block, len(down_block_types)),
            ("Transformer层数(transformer_layers_per_block)", transformer_layers_per_block, len(down_block_types))
        ]

        for name, value, target_len in param_config:
            if isinstance(value, int):
                original = value
                expanded = (value,) * target_len
                print(f"🔧 扩展 {name}:")
                print(f"   → 原始值: {original} → 扩展后: {expanded}")
                locals()[name.split('(')[0].strip()] = expanded  # 更新变量
            elif isinstance(value, (list, tuple)):
                print(f"✅ {name} 已为序列类型: {value}")

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
            
        # 类别嵌入连接处理
        print("\n[阶段3] 时间嵌入维度调整")
        if class_embeddings_concat:
            print("🌀 启用类别嵌入连接 (class_embeddings_concat=True)")
            print(f"   → 基础时间维度: {time_embed_dim} → 块时间维度: {time_embed_dim*2}")
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            print("⚙️ 使用标准时间嵌入维度")
            blocks_time_embed_dim = time_embed_dim
        print(f"✅ 最终块时间嵌入维度: {blocks_time_embed_dim}")

        # 打印最终配置
        print("\n📋 统一化后参数列表:")
        print(f"   → num_attention_heads: {num_attention_heads}")
        print(f"   → attention_head_dim: {attention_head_dim}")
        print(f"   → cross_attention_dim: {cross_attention_dim}")
        print(f"   → layers_per_block: {layers_per_block}")
        print(f"   → transformer_layers_per_block: {transformer_layers_per_block}")
        print(f"   → blocks_time_embed_dim: {blocks_time_embed_dim}")


#    def ok32424():
        print("\n[UNet Block Construction] 开始构建网络块")
        
        # 下采样块构建
        print(f"\n▼▼▼ 下采样块构建 ({len(down_block_types)}个) ▼▼▼")
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            print(f"\n[下采样块 {i+1}/{len(down_block_types)}] 类型: {down_block_type}")
            
            # 通道数配置
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            print(f"🔧 通道配置:")
            print(f"   → 输入通道: {input_channel}")
            print(f"   → 输出通道: {output_channel}")
            print(f"   → 是否最终块: {'是' if is_final_block else '否'}")

            # 获取下采样块参数
            print(f"\n⚙️ 块参数详情:")
            print(f"   ├─ 残差层数: {layers_per_block[i]}")
            print(f"   ├─ Transformer层数: {transformer_layers_per_block[i]}")
            print(f"   ├─ 时间嵌入维度: {blocks_time_embed_dim}")
            print(f"   ├─ 交叉注意力维度: {cross_attention_dim[i]}")
            print(f"   ├─ 注意力头数: {num_attention_heads[i]}")
            print(f"   ├─ 注意力头维度: {attention_head_dim[i] or '自动'}")
            print(f"   └─ 仅交叉注意力: {only_cross_attention[i]}")

            # 构建下采样块
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)
            print(f"✅ 块构建完成 → 模块结构: {down_block.__class__.__name__}")

        # 中间块构建
        print(f"\n▲▲▲ 中间块构建 ▲▲▲")
        print(f"🔧 配置参数:")
        print(f"   → 类型: {mid_block_type}")
        print(f"   → 输入通道: {block_out_channels[-1]}")
        print(f"   → 时间嵌入维度: {blocks_time_embed_dim}")
        print(f"   → 交叉注意力维度: {cross_attention_dim[-1]}")
        print(f"   → 注意力头数: {num_attention_heads[-1]}")
        print(f"   → 仅交叉注意力: {mid_block_only_cross_attention}")

        self.mid_block = get_mid_block(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=dropout,
        )
        print(f"✅ 中间块构建完成 → 模块结构: {self.mid_block.__class__.__name__}")                              


        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

        self._set_pos_net_if_use_gligen(attention_type=attention_type, cross_attention_dim=cross_attention_dim)

    def _check_config(
        self,
        down_block_types: Tuple[str],
        up_block_types: Tuple[str],
        only_cross_attention: Union[bool, Tuple[bool]],
        block_out_channels: Tuple[int],
        layers_per_block: Union[int, Tuple[int]],
        cross_attention_dim: Union[int, Tuple[int]],
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
        reverse_transformer_layers_per_block: bool,
        attention_head_dim: int,
        num_attention_heads: Optional[Union[int, Tuple[int]]],
    ):
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        print("\n=== 时间嵌入层初始化 ===")
        print(f"时间嵌入类型: {time_embedding_type}")
        print(f"输入参数: time_embedding_dim={time_embedding_dim}, block_out_channels[0]={block_out_channels[0]}")

        if time_embedding_type == "fourier":
            print("[分支] 选择 Fourier 嵌入")
            # 计算嵌入维度
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            print(f"    |-- 计算 time_embed_dim: {time_embed_dim} (原始输入: {time_embedding_dim})")
            
            # 奇偶校验
            if time_embed_dim % 2 != 0:
                error_msg = f"`time_embed_dim` 应为偶数，但得到 {time_embed_dim}"
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            else:
                print(f"    |-- 维度验证通过 (偶数)")

            # 初始化高斯傅里叶投影
            print(f"    |-- 初始化 GaussianFourierProjection:")
            print(f"        |-- 输出维度: {time_embed_dim // 2}")
            print(f"        |-- flip_sin_to_cos: {flip_sin_to_cos}")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, 
                set_W_to_weight=False, 
                log=False, 
                flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
            print(f"    |-- timestep_input_dim 设置为: {timestep_input_dim}")

        elif time_embedding_type == "positional":
            print("[分支] 选择 positional 嵌入")
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
            print(f"    |-- 计算 time_embed_dim: {time_embed_dim} (原始输入: {time_embedding_dim})")
            
            # 初始化位置编码
            print(f"    |-- 初始化 Timesteps:")
            print(f"        |-- 通道数: {block_out_channels[0]}")
            print(f"        |-- flip_sin_to_cos: {flip_sin_to_cos}")
            print(f"        |-- freq_shift: {freq_shift}")
            self.time_proj = Timesteps(
                block_out_channels[0], 
                flip_sin_to_cos, 
                freq_shift
            )
            timestep_input_dim = block_out_channels[0]
            print(f"    |-- timestep_input_dim 设置为: {timestep_input_dim}")

        else:
            error_msg = f"无效的时间嵌入类型: {time_embedding_type} (允许值: 'fourier' 或 'positional')"
            print(f"[错误] {error_msg}")
            raise ValueError(error_msg)

        print(f"最终配置: time_proj={self.time_proj.__class__.__name__}")
        print(f"         timestep_input_dim={timestep_input_dim}")


        return time_embed_dim, timestep_input_dim

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ):
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim_type`: {encoder_hid_dim_type} must be None, 'text_proj', 'text_image_proj', or 'image_proj'."
            )
        else:
            self.encoder_hid_proj = None

    def _set_class_embedding(
        self,
        class_embed_type: Optional[str],
        act_fn: str,
        num_class_embeds: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
        timestep_input_dim: int,
    ):
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

    def _set_add_embedding(
        self,
        addition_embed_type: str,
        addition_embed_type_num_heads: int,
        addition_time_embed_dim: Optional[int],
        flip_sin_to_cos: bool,
        freq_shift: float,
        cross_attention_dim: Optional[int],
        encoder_hid_dim: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
    ):
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(
                f"`addition_embed_type`: {addition_embed_type} must be None, 'text', 'text_image', 'text_time', 'image', or 'image_hint'."
            )

    def _set_pos_net_if_use_gligen(self, attention_type: str, cross_attention_dim: int):
        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, (list, tuple)):
                positive_len = cross_attention_dim[0]

            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = GLIGENTextBoundingboxProjection(
                positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type
            )

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        print("\n=== 时间步处理阶段 ===")
        print(f"输入 timestep 原始类型: {type(timestep)}, 值: {timestep}")

        # 类型转换分支
        if not torch.is_tensor(timesteps):
            print("[1] 检测到非张量输入，开始转换")
            is_mps = sample.device.type == "mps"
            print(f"    |-- 设备类型: {sample.device}, 是否 MPS: {is_mps}")

            # 确定数据类型
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
                print(f"    |-- 浮点类型转换: {timestep} -> {dtype}")
            else:
                dtype = torch.int32 if is_mps else torch.int64
                print(f"    |-- 整数类型转换: {timestep} -> {dtype}")
            
            # 创建张量
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            print(f"    |-- 生成张量: shape={timesteps.shape}, dtype={timesteps.dtype}, device={timesteps.device}")
        else:
            print("[1] 输入已是张量，原始形状:", timesteps.shape)

        # 标量张量处理
        if len(timesteps.shape) == 0:
            print("[2] 处理标量张量 (shape=[])")
            timesteps = timesteps[None].to(sample.device)
            print(f"    |-- 升维后形状: {timesteps.shape}, device: {timesteps.device}")

        # 广播到 batch 维度
        batch_size = sample.shape[0]
        print(f"[3] 广播前形状: {timesteps.shape}, 目标 batch_size: {batch_size}")
        timesteps = timesteps.expand(batch_size)
        print(f"    |-- 广播后形状: {timesteps.shape}")

        # 生成时间嵌入
        print("[4] 调用 time_proj 生成嵌入")
        t_emb = self.time_proj(timesteps)
        print(f"    |-- time_proj 输出形状: {t_emb.shape}, dtype={t_emb.dtype}")

        # 类型对齐
        print(f"[5] 对齐数据类型 (sample.dtype={sample.dtype})")
        t_emb = t_emb.to(dtype=sample.dtype)
        print(f"    |-- 最终 t_emb: shape={t_emb.shape}, dtype={t_emb.dtype}")

        return t_emb

    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb

    def get_aug_embed(
        self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        aug_emb = None
        print("\n=== 条件嵌入生成 ===")
        print(f"当前模式: {self.config.addition_embed_type}")
        print(f"输入参数检查: encoder_hidden_states.shape={encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
        # print(f"              added_cond_kwargs.keys()={list(added_cond_kwargs.keys())}")

        if self.config.addition_embed_type == "text":
            print("[分支] 纯文本条件模式")
            aug_emb = self.add_embedding(encoder_hidden_states)
            print(f"    |-- 直接使用 encoder_hidden_states 生成嵌入")
            print(f"    |-- aug_emb.shape: {aug_emb.shape}, dtype={aug_emb.dtype}")

        elif self.config.addition_embed_type == "text_image":
            print("[分支] 图文混合条件模式 (Kandinsky 2.1)")
            # 图像嵌入检查
            if "image_embeds" not in added_cond_kwargs:
                error_msg = f"缺失必要参数 'image_embeds'，当前参数: {list(added_cond_kwargs.keys())}"
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            
            # 获取嵌入
            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embs", encoder_hidden_states)
            print(f"    |-- 图像嵌入形状: {image_embs.shape}, 文本嵌入形状: {text_embs.shape}")
            
            # 生成联合嵌入
            aug_emb = self.add_embedding(text_embs, image_embs)
            print(f"    |-- 联合嵌入结果: shape={aug_emb.shape}, 均值={aug_emb.mean().item():.4f}")

        elif self.config.addition_embed_type == "text_time":
            print("[分支] 文本时间混合模式 (SDXL)")
            # 文本嵌入检查
            if "text_embeds" not in added_cond_kwargs:
                error_msg = f"缺失必要参数 'text_embeds'，当前参数: {list(added_cond_kwargs.keys())}"
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            text_embeds = added_cond_kwargs.get("text_embeds")
            print(f"    |-- 文本条件嵌入形状: {text_embeds.shape}")

            # 时间ID检查
            if "time_ids" not in added_cond_kwargs:
                error_msg = f"缺失必要参数 'time_ids'，当前参数: {list(added_cond_kwargs.keys())}"
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            time_ids = added_cond_kwargs.get("time_ids")
            print(f"    |-- 原始时间ID形状: {time_ids.shape}")

            # 时间嵌入处理
            time_embeds = self.add_time_proj(time_ids.flatten())
            print(f"    |-- 展平时间ID后: shape={time_ids.flatten().shape}")
            print(f"    |-- 投影后时间嵌入: shape={time_embeds.shape}")

            # 形状重塑
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            print(f"    |-- 重塑后时间嵌入: batch={text_embeds.shape[0]}, shape={time_embeds.shape}")

            # 拼接文本与时间
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            print(f"    |-- 拼接后嵌入: shape={add_embeds.shape} (文本 {text_embeds.shape[-1]} + 时间 {time_embeds.shape[-1]})")

            # 类型对齐
            add_embeds = add_embeds.to(emb.dtype)
            print(f"    |-- 对齐数据类型: {emb.dtype} → {add_embeds.dtype}")

            # 最终嵌入生成
            aug_emb = self.add_embedding(add_embeds)
            print(f"    |-- 最终条件嵌入: shape={aug_emb.shape}, 均值={aug_emb.mean().item():.4f}")     


        elif self.config.addition_embed_type == "image":
            print("[分支] 纯图像条件模式 (Kandinsky 2.2)")
            # 图像嵌入检查
            if "image_embeds" not in added_cond_kwargs:
                error_msg = (
                    f"缺失必要参数 'image_embeds'，当前可用参数: {list(added_cond_kwargs.keys())}\n"
                    f"提示: 请确保在调用时传入图像嵌入张量，例如 `pipe(..., added_cond_kwargs={'image_embeds': image_embeds})`"
                )
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            
            # 获取图像嵌入
            image_embs = added_cond_kwargs.get("image_embeds")
            print(f"    |-- 图像嵌入形状: {image_embs.shape}, dtype={image_embs.dtype}, 均值: {image_embs.mean().item():.4f}")
            
            # 生成嵌入
            aug_emb = self.add_embedding(image_embs)
            print(f"    |-- 增强嵌入结果: shape={aug_emb.shape}, 与时间嵌入兼容性检查: {aug_emb.shape == emb.shape}")

        elif self.config.addition_embed_type == "image_hint":
            print("[分支] 图像+控制提示模式 (Kandinsky 2.2 ControlNet)")
            # 参数存在性检查
            required_keys = ["image_embeds", "hint"]
            missing_keys = [key for key in required_keys if key not in added_cond_kwargs]
            if missing_keys:
                error_msg = (
                    f"缺失必要参数 {missing_keys}，当前可用参数: {list(added_cond_kwargs.keys())}\n"
                    f"提示: ControlNet 需要同时提供图像嵌入和空间提示（如边缘检测图）"
                )
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            
            # 获取参数
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            print(f"    |-- 图像嵌入形状: {image_embs.shape}, 均值: {image_embs.mean().item():.4f}")
            print(f"    |-- 控制提示形状: {hint.shape}, 通道数: {hint.shape[1]}, 值范围: [{hint.min().item():.3f}, {hint.max().item():.3f}]")
            
            # 生成联合嵌入
            aug_emb = self.add_embedding(image_embs, hint)
            print(f"    |-- 联合嵌入结果: shape={aug_emb.shape}, 与样本形状兼容性: {aug_emb.shape[2:] == sample.shape[2:]}")

        print("图像条件嵌入生成完成")

        return aug_emb

    def process_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        print("\n=== 编码器隐藏状态投影处理 ===")
        print(f"当前配置: encoder_hid_dim_type={self.config.encoder_hid_dim_type}")
        print(f"投影层状态: {'启用' if self.encoder_hid_proj else '未启用'}")

        # 文本投影分支
        if self.encoder_hid_proj and self.config.encoder_hid_dim_type == "text_proj":
            print("[分支] 纯文本投影 (text_proj)")
            print(f"输入文本编码形状: {encoder_hidden_states.shape}") if encoder_hidden_states is not None else print("警告: 文本编码为空")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
            print(f"投影后形状: {encoder_hidden_states.shape} | 数据类型: {encoder_hidden_states.dtype}")

        # 图文混合投影分支 (Kandinsky 2.1)
        elif self.encoder_hid_proj and self.config.encoder_hid_dim_type == "text_image_proj":
            print("[分支] 图文混合投影 (Kandinsky 2.1)")
            print(f"输入参数检查: added_cond_kwargs.keys()={list(added_cond_kwargs.keys())}")
            
            # 图像嵌入校验
            if "image_embeds" not in added_cond_kwargs:
                error_msg = (
                    "缺少必要参数 'image_embeds'\n"
                    f"当前可用参数: {list(added_cond_kwargs.keys())}\n"
                    "解决方案: 请确保调用时传递图像嵌入，例如:\n"
                    "pipe(..., added_cond_kwargs={'image_embeds': image_embeds})"
                )
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            
            image_embeds = added_cond_kwargs.get("image_embeds")
            print(f"图像嵌入属性: shape={image_embeds.shape} | 均值={image_embeds.mean().item():.4f}")
            print(f"文本编码输入形状: {encoder_hidden_states.shape}") if encoder_hidden_states is not None else print("警告: 文本编码为空")
            
            # 执行联合投影
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
            print(f"联合投影结果: shape={encoder_hidden_states.shape} | 标准差={encoder_hidden_states.std().item():.4f}")

        # 纯图像投影分支 (Kandinsky 2.2)
        elif self.encoder_hid_proj and self.config.encoder_hid_dim_type == "image_proj":
            print("[分支] 纯图像投影 (Kandinsky 2.2)")
            print(f"输入参数检查: added_cond_kwargs.keys()={list(added_cond_kwargs.keys())}")
            
            # 图像嵌入校验
            if "image_embeds" not in added_cond_kwargs:
                error_msg = (
                    "缺少必要参数 'image_embeds'\n"
                    f"当前可用参数: {list(added_cond_kwargs.keys())}\n"
                    "提示: 该配置需要单独的图像条件输入"
                )
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            
            image_embeds = added_cond_kwargs.get("image_embeds")
            print(f"图像嵌入属性: shape={image_embeds.shape} | 数据类型={image_embeds.dtype}")
            
            # 执行图像投影
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
            print(f"图像投影后形状: {encoder_hidden_states.shape} | 极值范围: [{encoder_hidden_states.min().item():.4f}, {encoder_hidden_states.max().item():.4f}]")

        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            print("\n=== IP适配器图像投影处理 ===")
            print(f"当前模式: {self.config.encoder_hid_dim_type}")
            print(f"文本编码投影层状态: {'已启用' if hasattr(self, 'text_encoder_hid_proj') else '未配置'}")

            # 图像嵌入参数强制检查
            if "image_embeds" not in added_cond_kwargs:
                error_msg = (
                    f"[配置冲突] 需要图像嵌入但未提供\n"
                    f"当前可用参数: {list(added_cond_kwargs.keys())}\n"
                    f"解决方案: 请通过 added_cond_kwargs 传递 image_embeds 参数\n"
                    f"示例: pipe(..., added_cond_kwargs={'{'}'image_embeds': image_embeds{'}'})"
                )
                print(f"[错误] {error_msg}")
                raise ValueError(error_msg)
            else:
                print(f"输入校验通过: 检测到 image_embeds (形状: {added_cond_kwargs['image_embeds'].shape})")

            # 文本编码投影处理
            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                print("\n[文本编码投影]")
                print(f"原始文本编码形状: {encoder_hidden_states.shape if encoder_hidden_states is not None else 'None'}")
                encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)
                print(f"投影后文本编码: {encoder_hidden_states.shape} | 数据类型: {encoder_hidden_states.dtype}")
            else:
                print("\n[文本编码投影] 跳过 (未配置 text_encoder_hid_proj)")

            # 图像嵌入投影处理
            print("\n[图像嵌入投影]")
            image_embeds = added_cond_kwargs.get("image_embeds")
            print(f"原始图像嵌入形状: {image_embeds.shape} | 均值: {image_embeds.mean().item():.4f}")

            image_embeds = self.encoder_hid_proj(image_embeds)
            print(f"投影后图像嵌入: {image_embeds.shape} | 值范围: [{image_embeds.min().item():.4f}, {image_embeds.max().item():.4f}]")

            # 多模态编码合并
            print("\n[多模态编码合并]")
            encoder_hidden_states = (encoder_hidden_states, image_embeds)
            print(f"合并后结构类型: {type(encoder_hidden_states)}")
            print(f"元组元素0 (文本): {encoder_hidden_states[0].shape}")
            print(f"元组元素1 (图像): {encoder_hidden_states[1].shape}")
            print(f"形状兼容性检查: 文本批次 {encoder_hidden_states[0].shape[0]} == 图像批次 {encoder_hidden_states[1].shape[0]} → {'通过' if encoder_hidden_states[0].shape[0] == encoder_hidden_states[1].shape[0] else '失败'}")

        return encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
    # def ok23423():
        print("\n[Upsample Check] 开始上采样尺寸检查")
        
        # 计算默认上采样因子
        default_overall_up_factor = 2 ** self.num_upsamplers
        print(f"🔍 计算总上采样因子: 2^{self.num_upsamplers} = {default_overall_up_factor}")
        
        # 初始化标志和尺寸
        forward_upsample_size = False
        upsample_size = None
        print(f"⚙️ 初始状态: forward_upsample_size={forward_upsample_size}, upsample_size={upsample_size}")

        # 检查各维度可除性
        print("\n[阶段1] 维度可除性验证")
        sample_dims = sample.shape[-2:]
        print(f"📏 输入样本最后两维: {sample_dims}")
        
        for idx, dim in enumerate(sample_dims):
            remainder = dim % default_overall_up_factor
            print(f"   → 维度 {['高度', '宽度'][idx]}: {dim} % {default_overall_up_factor} = {remainder}")
            
            if remainder != 0:
                print(f"❗ 检测到不可整除维度，标记需要调整上采样尺寸")
                forward_upsample_size = True
                break  # 任意维度不匹配即触发
        
        # 最终判断
        print("\n[阶段2] 上采样策略决策")
        if forward_upsample_size:
            print(f"🚨 启用强制尺寸调整 (forward_upsample_size=True)")


        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
    # def ok23432():
        print("\n[Sample Processing] 开始样本预处理流程")
        
        # 注意力掩码处理
        print("\n[阶段1] 注意力掩码转换")
        if attention_mask is not None:
            print(f"🔧 原始注意力掩码 shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            print(f"   → 转换后值域: [{attention_mask.min().item():.1f}, {attention_mask.max().item():.1f}]")
            attention_mask = attention_mask.unsqueeze(1)
            print(f"✅ 最终注意力掩码 shape: {attention_mask.shape}")
        else:
            print("⏭️ 未提供注意力掩码，跳过处理")

        # 编码器注意力掩码处理
        print("\n[阶段2] 编码器注意力掩码转换")
        if encoder_attention_mask is not None:
            print(f"🔧 原始编码器掩码 shape: {encoder_attention_mask.shape}, dtype: {encoder_attention_mask.dtype}")
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            print(f"   → 转换后值域: [{encoder_attention_mask.min().item():.1f}, {encoder_attention_mask.max().item():.1f}]")
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            print(f"✅ 最终编码器掩码 shape: {encoder_attention_mask.shape}")
        else:
            print("⏭️ 未提供编码器注意力掩码，跳过处理")

        # 输入中心化处理
        print("\n[阶段3] 输入归一化")
        if self.config.center_input_sample:
            print(f"⚙️ 执行输入中心化 (配置: center_input_sample=True)")
            print(f"   → 原始输入值域: [{sample.min().item():.2f}, {sample.max().item():.2f}]")
            sample = 2 * sample - 1.0
            print(f"✅ 中心化后值域: [{sample.min().item():.2f}, {sample.max().item():.2f}]")
        else:
            print("⏭️ 跳过输入中心化 (配置: center_input_sample=False)")

        # 时间嵌入处理
        print("\n[阶段4] 时间嵌入生成")
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        print(f"⏱️ 基础时间嵌入 shape: {t_emb.shape}")
        emb = self.time_embedding(t_emb, timestep_cond)
        print(f"✅ 增强时间嵌入 shape: {emb.shape} | dtype: {emb.dtype}")

        # 类别嵌入处理
        print("\n[阶段5] 类别嵌入处理")
        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            print(f"🏷️ 类别嵌入 shape: {class_emb.shape}")
            if self.config.class_embeddings_concat:
                print(f"🔀 拼接方式: concat(沿最后一维)")
                print(f"   → 原嵌入维度: {emb.shape[-1]}")
                print(f"   → 类别嵌入维度: {class_emb.shape[-1]}")
                emb = torch.cat([emb, class_emb], dim=-1)
                print(f"✅ 拼接后维度: {emb.shape[-1]}")
            else:
                print(f"➕ 合并方式: 元素相加")
                print(f"   → 原嵌入值域: [{emb.min().item():.2f}, {emb.max().item():.2f}]")
                emb = emb + class_emb
                print(f"✅ 合并后值域: [{emb.min().item():.2f}, {emb.max().item():.2f}]")
        else:
            print("⏭️ 未提供类别嵌入，跳过处理")



        # 1. 获取增强嵌入（aug_emb）
        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        print(f"[1] aug_emb 初始形状: {aug_emb.shape if aug_emb is not None else 'None'}, 类型: {type(aug_emb)}")

        # 处理 image_hint 类型的条件
        if self.config.addition_embed_type == "image_hint":
            print("[2] 进入 image_hint 分支")
            aug_emb, hint = aug_emb
            print(f"    |-- 拆分后 aug_emb 形状: {aug_emb.shape}, hint 形状: {hint.shape}")
            sample = torch.cat([sample, hint], dim=1)
            print(f"    |-- 拼接后 sample 形状: {sample.shape}")

        # 2. 时间嵌入与增强嵌入相加
        emb = emb + aug_emb if aug_emb is not None else emb
        print(f"[3] emb 相加后形状: {emb.shape}, 是否含 aug_emb: {aug_emb is not None}")

        # 3. 时间嵌入激活函数
        if self.time_embed_act is not None:
            print(f"[4] 应用激活函数: {self.time_embed_act.__class__.__name__}")
            emb = self.time_embed_act(emb)
            print(f"    |-- 激活后 emb 形状: {emb.shape}, 均值: {emb.mean().item():.4f}")

        # 4. 处理编码器隐藏状态
        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        print(f"[5] encoder_hidden_states 处理后形状: {encoder_hidden_states.shape}")

        # 5. 输入样本的预处理（卷积）
        print(f"[6] 输入卷积前 sample 形状: {sample.shape}")
        sample = self.conv_in(sample)
        print(f"    |-- 卷积后 sample 形状: {sample.shape}, 卷积层: {self.conv_in}")

        # 6. GLIGEN 位置网络处理
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            print("[7] 进入 GLIGEN 分支")
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            print(f"    |-- GLIGEN 参数类型: {type(gligen_args)}, 内容示例: { {k: v.shape for k, v in gligen_args.items()} }")
            gligen_output = self.position_net(**gligen_args)
            print(f"    |-- position_net 输出形状: {gligen_output['objs'].shape if isinstance(gligen_output, dict) else gligen_output.shape}")
            cross_attention_kwargs["gligen"] = {"objs": gligen_output}


        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        # 处理 cross_attention_kwargs 和 LoRA 缩放
        if cross_attention_kwargs is not None:
            print(f"[1] 处理 cross_attention_kwargs (原始 keys: {cross_attention_kwargs.keys()})")
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
            print(f"    |-- 弹出 LoRA scale 值: {lora_scale}, 剩余 keys: {cross_attention_kwargs.keys()}")
        else:
            lora_scale = 1.0
            print("[1] cross_attention_kwargs 为 None, 使用默认 LoRA scale=1.0")

        # PEFT 后端处理
        if USE_PEFT_BACKEND:
            print(f"[2] 应用 PEFT 后端 LoRA 缩放 (scale={lora_scale})")
            scale_lora_layers(self, lora_scale)
            num_scaled = sum(1 for n, _ in self.named_modules() if "lora" in n)  # 统计 LoRA 层数量
            print(f"    |-- 已缩放 {num_scaled} 个 LoRA 层")
        else:
            print("[2] 未启用 USE_PEFT_BACKEND，跳过 LoRA 缩放")

        # 检查 ControlNet/Adapter 条件
        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None
        print(f"[3] 条件检查: is_controlnet={is_controlnet}, is_adapter={is_adapter}")

        # Adapter 参数弃用警告处理
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            print("[4] 检测到弃用参数使用!")
            print(f"    |-- 原始 down_block_additional_residuals 类型: {type(down_block_additional_residuals)}")
            print(f"    |-- 参数形状示例: down_block[0].shape={down_block_additional_residuals[0].shape if len(down_block_additional_residuals)>0 else 'empty'}")
            
            # 执行参数转移
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True
            print(f"    |-- 已将 down_block 参数转移到 intrablock，is_adapter={is_adapter}")
            
            # 显示弃用警告
            deprecation_msg = "T2I should not use down_block_additional_residuals"
            print(f"    |-- 弃用警告: {deprecation_msg} (since v1.3.0)")


        down_block_res_samples = (sample,)
        print(f"[初始化] down_block_res_samples 初始长度: {len(down_block_res_samples)}, 首元素形状: {down_block_res_samples[0].shape}")

        for idx, downsample_block in enumerate(self.down_blocks):
            print(f"\n=== 处理下采样块 {idx} ({downsample_block.__class__.__name__}) ===")
            print(f"输入 sample 形状: {sample.shape}")
            
            # 检查是否为 CrossAttn 块
            has_cross_attn = hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention
            print(f"是否含交叉注意力: {has_cross_attn}, 是否适配器模式: {is_adapter}")
            
            additional_residuals = {}
            if has_cross_attn:
                # CrossAttn 块处理分支
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    residual = down_intrablock_additional_residuals[0]
                    print(f"[适配器] 准备注入残差，剩余残差数: {len(down_intrablock_additional_residuals)}, 当前残差形状: {residual.shape}")
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)
                
                # 执行下采样块前向
                print(f"调用 CrossAttnDownBlock2D 参数: encoder_hidden_states={encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                # 普通下采样块处理
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    residual = down_intrablock_additional_residuals[0]
                    print(f"[适配器] 添加残差，剩余残差数: {len(down_intrablock_additional_residuals)}, 残差形状: {residual.shape}")
                    sample += down_intrablock_additional_residuals.pop(0)
                    print(f"添加后 sample 形状: {sample.shape}, 均值变化: {sample.mean().item() - res_samples[0].mean().item():+.4f}")
            
            # 记录残差
            print(f"下采样块 {idx} 输出: sample 形状={sample.shape}, res_samples 长度={len(res_samples)}")
            down_block_res_samples += res_samples
            print(f"更新 down_block_res_samples 长度: {len(down_block_res_samples)}")


        # ControlNet 残差处理分支
        if is_controlnet:
            print(f"\n[ControlNet] 开始融合下采样残差 (原残差数量: {len(down_block_res_samples)})")
            print(f"    |-- down_block_additional_residuals 长度: {len(down_block_additional_residuals)}")
            
            new_down_block_res_samples = ()
            for i, (down_block_res_sample, down_block_additional_residual) in enumerate(
                zip(down_block_res_samples, down_block_additional_residuals)
            ):
                print(f"    |-- 处理第 {i} 个残差块")
                print(f"        |-- 原始残差形状: {down_block_res_sample.shape}")
                print(f"        |-- ControlNet 附加残差形状: {down_block_additional_residual.shape}")
                
                # 执行残差相加
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                print(f"        |-- 融合后残差形状: {down_block_res_sample.shape}, 均值变化: {down_block_res_sample.mean().item() - down_block_res_sample.mean().item():+.4f}")
                
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            
            down_block_res_samples = new_down_block_res_samples
            print(f"[ControlNet] 融合完成，新残差数量: {len(down_block_res_samples)}")

        # 中间块处理
        if self.mid_block is not None:
            print("\n[中间块] 输入 sample 形状:", sample.shape)
            
            # 判断是否含交叉注意力
            mid_has_cross_attn = hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention
            print(f"    是否含交叉注意力: {mid_has_cross_attn}")
            
            if mid_has_cross_attn:
                print("    调用带交叉注意力的 mid_block")
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                print("    调用普通 mid_block")
                sample = self.mid_block(sample, emb)
            
            print(f"[中间块] 输出 sample 形状: {sample.shape}")

            # T2I-Adapter-XL 残差注入
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                print("\n[适配器] 尝试中间块残差注入")
                residual = down_intrablock_additional_residuals[0]
                print(f"    |-- 剩余残差数量: {len(down_intrablock_additional_residuals)}, 当前残差形状: {residual.shape}")
                print(f"    |-- sample 形状: {sample.shape}, 是否匹配: {sample.shape == residual.shape}")
                
                if sample.shape == residual.shape:
                    sample += down_intrablock_additional_residuals.pop(0)
                    print("    |-- 成功注入残差，更新后 sample 均值:", sample.mean().item())
                else:
                    print("    |-- 形状不匹配，跳过注入")

                

        # ControlNet 中间残差处理
        if is_controlnet:
            print(f"\n[ControlNet] 注入中间块残差 (形状: {mid_block_additional_residual.shape})")
            print(f"    |-- 注入前 sample 均值: {sample.mean().item():.4f}")
            sample = sample + mid_block_additional_residual
            print(f"    |-- 注入后 sample 均值: {sample.mean().item():.4f}, 形状: {sample.shape}")

        # 上采样循环处理
        print(f"\n=== 进入上采样阶段，共 {len(self.up_blocks)} 个上采样块 ===")
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            print(f"\n[上采样块 {i}] 类型: {upsample_block.__class__.__name__}, 是否最终块: {is_final_block}")
            
            # 残差切片管理
            res_samples_len = len(upsample_block.resnets)
            res_samples = down_block_res_samples[-res_samples_len:]
            down_block_res_samples = down_block_res_samples[:-res_samples_len]
            print(f"    |-- 取 {res_samples_len} 个残差，剩余残差数: {len(down_block_res_samples)}")
            print(f"    |-- 当前残差形状列表: {[s.shape for s in res_samples]}")

            # 上采样尺寸传递
            upsample_size = None
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
                print(f"    |-- 非最终块，设置 upsample_size={upsample_size}")

            # 判断是否含交叉注意力
            has_cross_attn = hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention
            print(f"    |-- 是否含交叉注意力: {has_cross_attn}")

            # 执行上采样块前向
            if has_cross_attn:
                print(f"    |-- 调用带交叉注意力的上采样块，encoder_hidden_states 形状: {encoder_hidden_states.shape}")
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                print("    |-- 调用普通上采样块")
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
            print(f"    |-- 上采样块 {i} 输出形状: {sample.shape}")

        # 后处理步骤
        print("\n=== 进入后处理阶段 ===")
        if self.conv_norm_out:
            print(f"应用 conv_norm_out: {self.conv_norm_out}")
            sample = self.conv_norm_out(sample)
            print(f"    |-- 归一化后 sample 均值: {sample.mean().item():.4f}, 形状: {sample.shape}")
            
            print(f"应用 conv_act: {self.conv_act.__class__.__name__}")
            sample = self.conv_act(sample)
            print(f"    |-- 激活后 sample 均值: {sample.mean().item():.4f}")

        print(f"应用 conv_out: {self.conv_out} (输入通道: {self.conv_out.in_channels}, 输出通道: {self.conv_out.out_channels})")
        sample = self.conv_out(sample)
        print(f"后处理最终输出形状: {sample.shape}")

        # PEFT 后端清理
        if USE_PEFT_BACKEND:
            print("\n[PEFT] 清理 LoRA 缩放因子 (scale=1.0)")
            unscale_lora_layers(self, lora_scale)
            num_reset = sum(1 for n, _ in self.named_modules() if "lora" in n)
            print(f"    |-- 已重置 {num_reset} 个 LoRA 层")

        if not return_dict:
            print("\n返回非字典格式输出 (sample,)")
            return (sample,)
            
        return UNet2DConditionOutput(sample=sample)
