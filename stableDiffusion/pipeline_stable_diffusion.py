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
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from torch_utils import randn_tensor
from pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from pipeline_output import StableDiffusionPipelineOutput
from safety_checker import StableDiffusionSafetyChecker


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    """
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

    # def ok121():
        print("\n[Deprecation Handler] 开始检查调度器配置兼容性")
        
        # 检查 steps_offset 配置
        print("[步骤1] 检查 steps_offset 配置状态...")
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            print(f"⚠️ 检测到旧版 steps_offset 配置: {scheduler.config.steps_offset} (预期值应为 1)")
            
            # 构建弃用消息
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            print(f"📢 触发弃用警告: {deprecation_message}")
            
            # 执行配置更新
            print("🛠️ 开始自动更新 steps_offset 配置...")
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)
            print(f"✅ 配置更新完成 | 新 steps_offset: {scheduler.config.steps_offset}")
        else:
            print("[步骤1] steps_offset 配置符合要求，无需更新")
        
        # 检查 clip_sample 配置
        print("\n[步骤2] 检查 clip_sample 配置状态...")
        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            print(f"⚠️ 检测到旧版 clip_sample 配置: {scheduler.config.clip_sample} (预期值应为 False)")
            
            # 构建弃用消息
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            print(f"📢 触发弃用警告: {deprecation_message}")
            
            # 执行配置更新
            print("🛠️ 开始自动更新 clip_sample 配置...")
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)
            print(f"✅ 配置更新完成 | 新 clip_sample: {scheduler.config.clip_sample}")
        else:
            print("[步骤2] clip_sample 配置符合要求，无需更新")
    

    # def ok2342():
        print("\n[Safety Check] 开始安全检查流程")
        
        # 安全检查器校验
        print("[阶段1] 验证安全检查器配置...")
        if safety_checker is None and requires_safety_checker:
            print(f"⚠️ 检测到安全隐患: safety_checker=None | requires_safety_checker={requires_safety_checker}")
            warning_msg = (
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )
            print(f"📢 安全警告已记录: {warning_msg[:80]}...")  # 显示前80字符防止日志过长
        else:
            print("✅ 安全检查器配置符合要求")

        # 特征提取器依赖检查
        print("\n[阶段2] 验证特征提取器依赖...")
        if safety_checker is not None and feature_extractor is None:
            error_msg = (
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
            print(f"❌ 致命配置错误: {error_msg}")
            print(f"当前状态: safety_checker={safety_checker is not None} | feature_extractor={feature_extractor is None}")
            raise ValueError(error_msg)
        else:
            print("✅ 特征提取器依赖满足要求")

        # UNet版本检测
        print("\n[阶段3] 分析UNet配置")
        print("🔍 检查UNet版本兼容性...")
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        
        version_info = unet.config._diffusers_version if hasattr(unet.config, "_diffusers_version") else "未知版本"
        print(f"   → 当前UNet版本: {version_info}")
        print(f"   → 版本低于0.9.0.dev0? {'是' if is_unet_version_less_0_9_0 else '否'}")

        # 样本尺寸检测
        print("\n🔍 检查样本尺寸配置...")
        self._is_unet_config_sample_size_int = isinstance(unet.config.sample_size, int)
        sample_size_value = unet.config.sample_size if hasattr(unet.config, "sample_size") else "未定义"
        
        print(f"   → 样本尺寸类型: {type(unet.config.sample_size).__name__}" if hasattr(unet.config, "sample_size") 
            else "   → 样本尺寸: 未配置")
        print(f"   → 是否为整数类型? {'是' if self._is_unet_config_sample_size_int else '否'}")

        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and 
            self._is_unet_config_sample_size_int and 
            unet.config.sample_size < 64
        )
        print(f"   → 样本尺寸小于64? {'是' if is_unet_sample_size_less_64 else '否'} (当前值: {sample_size_value})")

    # def ok32532():
        print("\n[System Initialization] 开始系统初始化流程")

        # UNET配置弃用检查
        print("\n[阶段4] 执行UNET配置兼容性检查")
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            print("⚠️ 检测到不兼容的UNET配置组合：")
            print(f"   → UNET版本 < 0.9.0: {'是' if is_unet_version_less_0_9_0 else '否'}")
            print(f"   → 样本尺寸 < 64: {'是' if is_unet_sample_size_less_64 else '否'}")

            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            print(f"📜 弃用通知: {deprecation_message[:120]}...")  # 显示前120个字符
            
            print("🛠️ 开始自动更新UNET配置...")
            new_config = dict(unet.config)
            original_size = new_config.get("sample_size", "未配置")
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
            print(f"✅ UNET配置更新完成 | 样本尺寸: {original_size} → {unet.config.sample_size}")

        # 模块注册过程
        print("\n[阶段5] 注册核心模块")
        modules_to_register = {
            'vae': vae,
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'unet': unet,
            'scheduler': scheduler,
            'safety_checker': safety_checker,
            'feature_extractor': feature_extractor,
            'image_encoder': image_encoder
        }
        
        print("📦 正在注册以下模块:")
        for name, module in modules_to_register.items():
            status = "已启用" if module is not None else "未配置"
            print(f"   → {name.ljust(15)}: {status.ljust(8)} ({type(module).__name__})")
        
        self.register_modules(**modules_to_register)
        print("✅ 所有模块注册完成")

        # VAE缩放因子计算
        print("\n[阶段6] 计算VAE缩放因子")
        block_out_channels = self.vae.config.block_out_channels
        vae_scale_exp = len(block_out_channels) - 1
        self.vae_scale_factor = 2 ** vae_scale_exp
        print(f"   → Block通道数: {block_out_channels}")
        print(f"   → 计算表达式: 2^{vae_scale_exp} = {self.vae_scale_factor}")
        print(f"🔧 VAE缩放因子已设置为: {self.vae_scale_factor}")

        # 图像处理器初始化
        print("\n[阶段7] 初始化图像处理器")
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        print(f"🖼️ 图像处理器初始化完成 | 类型: {type(self.image_processor).__name__}")
        print(f"   → 使用缩放因子: {self.image_processor.vae_scale_factor}")

        # 安全检测配置
        print("\n[阶段8] 写入最终配置")
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        print(f"⚙️ 安全检测需求已固化: {requires_safety_checker}")
        
        print("\n[System Initialization] 系统初始化流程完成 ✅\n")


    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
    # def ok32432():
        print("\n[Text Processing] 开始文本处理流程")
        
        # LoRA缩放处理
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            print(f"\n🔄 动态调整LoRA缩放 (比例: {lora_scale})")
            self._lora_scale = lora_scale
            if not USE_PEFT_BACKEND:
                print("   → 使用原生LoRA调整方法")
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                print("   → 使用PEFT后端调整方法")
                scale_lora_layers(self.text_encoder, lora_scale)
            print(f"✅ 文本编码器LoRA层已更新")
        elif lora_scale is not None:
            print(f"\n⚠️ 忽略LoRA缩放请求 (当前模型不支持LoRA)")

        # 批次大小确定
        print("\n📦 确定批量大小")
        if prompt is not None:
            input_type = "str" if isinstance(prompt, str) else "list"
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            print(f"   → 来源: 文本输入 ({input_type}) → 批量大小: {batch_size}")
        else:
            batch_size = prompt_embeds.shape[0]
            print(f"   → 来源: 预生成提示嵌入 → 批量大小: {batch_size}")
        print(f"✅ 最终批量大小: {batch_size}")

        # 文本嵌入处理
        if prompt_embeds is None:
            print("\n🔡 初始化文本嵌入生成")
            if isinstance(self, TextualInversionLoaderMixin):
                print("🌀 检测到文本反演加载器，进行多向量标记处理")
                original_prompt = prompt
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if prompt != original_prompt:
                    print(f"   → 文本转换: '{original_prompt}' → '{prompt}'")
            
            print(f"\n🔠 执行文本标记化 (model_max_length={self.tokenizer.model_max_length})")
            print(f"   → Padding策略: max_length")
            # print(f"   → 截断策略: {'启用' if truncation else '禁用'}")
            
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            print(f"✅ 标记化结果: shape={text_input_ids.shape} | dtype={text_input_ids.dtype}")

            # 截断验证
            print("\n🔍 验证输入截断情况")
            untruncated_ids = self.tokenizer(
                prompt, 
                padding="longest", 
                return_tensors="pt"
            ).input_ids



        # def ok324324():
            print("\n[Text Encoding] 开始文本编码流程")
            
            # 文本截断警告处理
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                print("\n⚠️ 检测到输入截断")
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                print(f"   → 模型最大长度: {self.tokenizer.model_max_length} tokens")
                print(f"   → 被截断内容: {removed_text}")
                print(f"   → 原始长度: {untruncated_ids.shape[-1]} | 截断后长度: {text_input_ids.shape[-1]}")

            # 注意力掩码配置
            print("\n[阶段1] 注意力机制配置")
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
                print(f"✅ 启用注意力掩码 | shape: {attention_mask.shape} | dtype: {attention_mask.dtype}")
            else:
                attention_mask = None
                print("⚙️ 未配置注意力掩码")

            # CLIP层跳过处理
            print("\n[阶段2] 文本编码执行")
            if clip_skip is None:
                print(f"🌀 标准CLIP编码 (clip_skip=None)")
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                print(f"⏭️ 跳过最后{clip_skip}个CLIP层")
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                print(f"   → 获取第{- (clip_skip + 1)}层隐藏状态")
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                print(f"   → 层归一化前形状: {prompt_embeds.shape}")
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)
            


                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                # prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                # prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

    # def ok3243242():
        print("\n[Embedding Preparation] 开始嵌入预处理")
        
        # 确定嵌入数据类型
        print("\n[阶段1] 数据类型验证")
        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
            print(f"🔍 从text_encoder获取数据类型: {prompt_embeds_dtype}")
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
            print(f"🔍 从unet获取数据类型: {prompt_embeds_dtype}")
        else:
            prompt_embeds_dtype = prompt_embeds.dtype
            print(f"⚠️ 从嵌入本身推断数据类型: {prompt_embeds_dtype}")
        
        print(f"⚙️ 转换嵌入到 {prompt_embeds_dtype} 类型 | 设备: {device}")
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        print(f"✅ 当前嵌入设备: {prompt_embeds.device} | dtype: {prompt_embeds.dtype}")

        # 扩展嵌入维度
        print("\n[阶段2] 嵌入扩展")
        original_shape = prompt_embeds.shape
        print(f"📦 原始形状: (batch_size={original_shape[0]}, seq_len={original_shape[1]}, dim={original_shape[2]})")
        
        print(f"🔄 按每提示生成数扩展: {num_images_per_prompt}x")
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(original_shape[0] * num_images_per_prompt, original_shape[1], -1)
        
        print(f"✅ 扩展后形状: {prompt_embeds.shape}")

        # 处理负向提示
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            print("\n[阶段3] 生成无条件嵌入")
            print(f"🔧 分类器自由引导比例: {self.guidance_scale}")
            
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
                print(f"⚙️ 使用空负向提示 (batch_size={batch_size})")
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                print(f"⚙️ 单文本负向提示扩展至批次大小 {batch_size}")
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                error_msg = (f"❌ 批次大小不匹配: 负向提示数量 {len(negative_prompt)} "
                        f"≠ 正向提示数量 {batch_size}")
                print(error_msg)
                raise ValueError(error_msg)
            else:
                uncond_tokens = negative_prompt
                print(f"✅ 有效负向提示数量: {len(uncond_tokens)}")



        # def ok234324():
            print("\n[Negative Prompt Processing] 开始负向提示处理")
            
            # 文本反演处理
            if isinstance(self, TextualInversionLoaderMixin):
                print("\n🌀 检测到文本反演加载器")
                original_uncond = uncond_tokens
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)
                if uncond_tokens != original_uncond:
                    print(f"   → 转换特殊标记: {original_uncond} → {uncond_tokens}")
            else:
                print("\n⚙️ 未启用文本反演处理")

            # 标记化处理
            max_length = prompt_embeds.shape[1]
            print(f"\n🔠 负向提示标记化 (max_length={max_length})")
            print(f"   → 输入token数量: {len(uncond_tokens)}条提示")
            print(f"   → Padding策略: max_length ({max_length} tokens)")
            
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            print(f"✅ 标记化结果: input_ids形状={uncond_input.input_ids.shape}")

            # 注意力掩码配置
            print("\n🎭 注意力机制配置")
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
                print(f"   → 启用注意力掩码 | 设备: {attention_mask.device} | 类型: {attention_mask.dtype}")
            else:
                attention_mask = None
                print("⚙️ 未配置注意力掩码")

            # 文本编码
            print("\n🧠 执行负向提示编码")
            print(f"   → 输入设备: {device}")
            print(f"   → 输入形状: {uncond_input.input_ids.shape}")
            
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            
            print(f"\n✅ 负向嵌入生成完成:")
            print(f"   → 输出形状: {negative_prompt_embeds.shape}")
            print(f"   → 数据类型: {negative_prompt_embeds.dtype}")
            print(f"   → 均值: {negative_prompt_embeds.mean().item():.4f} ± {negative_prompt_embeds.std().item():.4f}")

            # print("\n[Negative Prompt Processing] 处理完成 ✅\n")
            # return negative_prompt_embeds

    # def ok32432():
        print("\n[CFG Preparation] 开始分类器自由引导准备")
        
        if do_classifier_free_guidance:
            print("\n[阶段1] 负向提示嵌入处理")
            seq_len = negative_prompt_embeds.shape[1]
            print(f"📏 原始负向嵌入形状: {negative_prompt_embeds.shape} (seq_len={seq_len})")
            
            # 数据类型转换
            print(f"⚙️ 数据类型对齐: {negative_prompt_embeds.dtype} → {prompt_embeds_dtype}")
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            print(f"✅ 当前设备: {negative_prompt_embeds.device} | dtype: {negative_prompt_embeds.dtype}")
            
            # 嵌入扩展
            print(f"\n🔄 扩展负向嵌入 (每提示生成数: {num_images_per_prompt})")
            print(f"   → 原始批次大小: {batch_size}")
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            print(f"✅ 扩展后形状: {negative_prompt_embeds.shape}")
        else:
            print("\n⏭️ 跳过CFG准备 (未启用分类器自由引导)")

        if self.text_encoder is not None:
            print("\n[阶段2] LoRA层调整")
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                print(f"🔧 恢复LoRA原始比例 (当前scale={lora_scale})")
                # print(f"   → 文本编码器层数: {len(self.text_encoder.layers)}")
                unscale_lora_layers(self.text_encoder, lora_scale)
                print("✅ LoRA层已恢复默认比例")
            else:
                print("⚙️ 跳过LoRA调整 (不满足条件)")



        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    # def ok23432():
        print("\n[Latent Generation] 开始潜在变量初始化")
        
        # 计算潜在变量形状
        print("\n[阶段1] 形状计算")
        h = int(height) // self.vae_scale_factor
        w = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, h, w)
        print(f"✅ 潜在空间形状: {shape}")
        print(f"   → 原始分辨率: {height}x{width}")
        print(f"   → VAE缩放因子: {self.vae_scale_factor}")
        print(f"   → 缩放后分辨率: {h}x{w}")
        print(f"   → 通道数: {num_channels_latents}")
        print(f"   → 批次大小: {batch_size}")

        # 验证生成器配置
        print("\n[阶段2] 随机生成器验证")
        if isinstance(generator, list):
            print(f"🔍 检测到生成器列表 (长度: {len(generator)})")
            if len(generator) != batch_size:
                print(f"❌ 不匹配: 生成器数量({len(generator)}) ≠ 批次大小({batch_size})")
                raise ValueError(
                    f"生成器数量与批次大小不匹配: {len(generator)} vs {batch_size}"
                )
            else:
                print("✅ 生成器列表与批次大小匹配")
        else:
            print(f"⚙️ 使用单一生成器 (类型: {type(generator).__name__})")

        # 初始化潜在变量
        print("\n[阶段3] 噪声生成")
        if latents is None:
            print("🔄 生成新潜在变量")
            print(f"   → 设备: {device}")
            print(f"   → 数据类型: {dtype}")
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            print(f"✅ 初始潜在变量统计:")
            print(f"   → 形状: {latents.shape}")
            print(f"   → 均值: {latents.mean().item():.4f}")
            print(f"   → 标准差: {latents.std().item():.4f}")
            print(f"   → 值域: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
        else:
            print("⚡ 使用预生成潜在变量")
            print(f"   → 输入形状: {latents.shape}")
            print(f"   → 原始设备: {latents.device} → 目标设备: {device}")
            latents = latents.to(device)
            print(f"✅ 迁移后潜在变量设备: {latents.device}")

        # 噪声缩放
        print("\n[阶段4] 噪声缩放")
        init_noise_sigma = self.scheduler.init_noise_sigma
        print(f"🔧 应用初始噪声缩放系数: {init_noise_sigma:.4f}")
        print(f"   → 缩放前均值: {latents.mean().item():.4f}")
        latents = latents * init_noise_sigma
        print(f"✅ 缩放后统计:")
        print(f"   → 均值: {latents.mean().item():.4f}")
        print(f"   → 标准差: {latents.std().item():.4f}")
        print(f"   → 值域: [{latents.min().item():.4f}, {latents.max().item():.4f}]")

        print("\n[Latent Generation] 初始化完成 ✅\n")
        return latents


    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

    # def ok23432():
        print("\n[Callbacks Setup] 开始回调函数配置")
        
        # 处理旧版回调参数
        print("\n[阶段1] 处理回调参数弃用")
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        deprecated_params = [
            ("callback", kwargs.get("callback")),
            ("callback_steps", kwargs.get("callback_steps"))
        ]
        
        for param, value in deprecated_params:
            if value is not None:
                print(f"⚠️ 检测到弃用参数 {param} = {value}")
                print(f"   → 替代方案: 使用 callback_on_step_end 参数")
                deprecate(
                    param,
                    "1.0.0",
                    f"Passing `{param}` 已弃用，请使用 `callback_on_step_end`"
                )
                kwargs.pop(param)
        print("✅ 弃用参数处理完成")

        # 处理新版回调配置
        print("\n[阶段2] 配置新版回调系统")
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            print(f"✅ 检测到有效回调处理器: {type(callback_on_step_end).__name__}")
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
            print(f"   → 注册的输入参数: {callback_on_step_end_tensor_inputs}")
        else:
            print("⚙️ 未配置回调处理器")

        # 设置默认分辨率
        print("\n[阶段3] 图像分辨率设置")
        if not height or not width:
            print("🔍 自动获取默认分辨率...")
            sample_size = self.unet.config.sample_size
            
            # 判断样本尺寸类型
            size_type = "整数" if self._is_unet_config_sample_size_int else "元组"
            print(f"   → UNet配置样本尺寸: {sample_size} ({size_type})")
            
            base_height = sample_size if self._is_unet_config_sample_size_int else sample_size[0]
            base_width = sample_size if self._is_unet_config_sample_size_int else sample_size[1]
            print(f"   → 基础分辨率: {base_height}x{base_width}")
            
            # 应用VAE缩放因子
            print(f"   → VAE缩放因子: {self.vae_scale_factor}")
            height = base_height * self.vae_scale_factor
            width = base_width * self.vae_scale_factor
            print(f"✅ 最终默认分辨率: {height}x{width}")
        else:
            print(f"✅ 使用自定义分辨率: {height}x{width}")

        # print("\n[Callbacks Setup] 配置完成 ✅\n")

        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
    # def ok32432():
        print("\n[Pipeline Setup] 开始推理流程初始化")
        
        # 输入验证阶段
        print("\n[阶段1] 输入参数验证")
        print("🔍 执行输入完整性检查...")
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        print("✅ 所有输入参数验证通过")

        # 参数设置阶段
        print("\n[阶段2] 配置核心参数")
        param_config = [
            ("guidance_scale", guidance_scale),
            ("guidance_rescale", guidance_rescale),
            ("clip_skip", clip_skip),
            ("cross_attention_kwargs", cross_attention_kwargs)
        ]
        for name, value in param_config:
            print(f"   → {name.ljust(25)}: {str(value).ljust(15)} ({type(value).__name__})")
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False
        print("⚙️ 中断标志初始化: False")

        # 批处理设置
        print("\n[阶段3] 批处理配置")
        if prompt is not None:
            input_type = "str" if isinstance(prompt, str) else "list"
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            print(f"📦 文本提示类型: {input_type} → 批量大小: {batch_size}")
        else:
            batch_size = prompt_embeds.shape[0]
            print(f"📦 使用预先生成的提示嵌入 → 批量大小: {batch_size}")
        print(f"🔧 计算设备: {self._execution_device}")
        device = self._execution_device
        # 注意力机制配置
        print("\n[阶段4] 注意力参数设置")
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs else None
        print(f"🔗 LoRA缩放因子: {lora_scale or '未启用'}")
        print(f"📌 CLIP跳过层数: {self.clip_skip}")

        # 提示编码过程
        print("\n[阶段5] 文本提示编码")
        print(f"📝 正向提示数量: {len(prompt) if isinstance(prompt, list) else 1}")
        print(f"📝 负向提示数量: {len(negative_prompt) if isinstance(negative_prompt, list) else 1}")
        print(f"🔄 分类器自由引导: {'启用' if self.do_classifier_free_guidance else '禁用'}")
        
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        
        print("✅ 编码结果:")
        print(f"   → 正向嵌入形状: {tuple(prompt_embeds.shape)}")
        print(f"   → 负向嵌入形状: {tuple(negative_prompt_embeds.shape)}")
        print(f"   → 每提示生成数: {num_images_per_prompt}")

        # print("\n[Pipeline Setup] 初始化完成 ✅\n")


        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
    # def ok23432():
        print("\n[Generation Setup] 开始生成准备流程")
        
        # 分类器自由引导处理
        print("\n[阶段1] 分类器自由引导处理")
        print(f"🔧 引导模式: {'启用' if self.do_classifier_free_guidance else '禁用'}")
        if self.do_classifier_free_guidance:
            print(f"📐 合并嵌入前形状:")
            print(f"   → 负向提示嵌入: {tuple(negative_prompt_embeds.shape)}")
            print(f"   → 正向提示嵌入: {tuple(prompt_embeds.shape)}")
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            print(f"✅ 合并后提示嵌入形状: {tuple(prompt_embeds.shape)}")
        else:
            print("⚙️ 跳过提示嵌入合并步骤")

        # IP适配器图像处理
        print("\n[阶段2] IP适配器图像嵌入")
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            print("🖼️ 检测到图像输入:")
            print(f"   → 输入类型: {'图像文件' if ip_adapter_image else '预生成嵌入'}")
            print(f"   → 批量大小: {batch_size * num_images_per_prompt}")
            print(f"   → 设备: {device}")
            
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
            print(f"✅ 生成图像嵌入形状: {tuple(image_embeds.shape)}")
        else:
            print("⚙️ 未检测到IP适配器输入")

        # 时间步准备
        print("\n[阶段3] 时间步配置")
        print(f"🔧 输入参数:")
        print(f"   → 推理步数: {num_inference_steps or '自动'}")
        print(f"   → 自定义时间步: {timesteps[:3] if timesteps is not None else '无'}...")
        
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        print(f"✅ 最终时间步参数:")
        print(f"   → 总推理步数: {num_inference_steps}")
        print(f"   → 时间步形状: {tuple(timesteps.shape)}")
        print(f"   → 时间步范围: [{timesteps[0].item():.1f}, {timesteps[-1].item():.1f}]")
        num_channels_latents = self.unet.config.in_channels
        # 潜在变量初始化
        print("\n[阶段4] 潜在空间初始化")
        print(f"📦 潜在变量参数:")
        print(f"   → 输入通道数: {num_channels_latents}")
        print(f"   → 目标分辨率: {height}x{width}")
        print(f"   → 数据类型: {prompt_embeds.dtype}")
        
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        print(f"✅ 潜在变量生成结果:")
        print(f"   → 形状: {tuple(latents.shape)}")
        print(f"   → 均值: {latents.mean().item():.4f}")
        print(f"   → 标准差: {latents.std().item():.4f}")
        if generator is not None:
            print(f"🔧 使用生成器设备: {generator.device}")

        # 额外参数准备
        print("\n[阶段5] 扩散过程参数")
        print(f"🔧 噪声调度参数eta: {eta}")
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        print(f"✅ 额外参数内容: {list(extra_step_kwargs.keys())}")

        # IP适配器条件参数
        print("\n[阶段6] 图像条件参数装配")
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )
        if added_cond_kwargs:
            print(f"📦 添加图像条件参数:")
            print(f"   → 嵌入形状: {tuple(added_cond_kwargs['image_embeds'].shape)}")
            print(f"   → 设备: {added_cond_kwargs['image_embeds'].device}")
        else:
            print("⚙️ 无附加图像条件参数")

        # print("\n[Generation Setup] 准备流程完成 ✅\n")


        # 6.2 Optionally get Guidance Scale Embedding
    # def ok23423():
        print("\n[Denoising Loop] 开始去噪迭代流程")
        
        # 时间条件投影处理
        print("\n[阶段1] 时间条件设置")
        if self.unet.config.time_cond_proj_dim is not None:
            print(f"⏱️ 生成引导规模条件嵌入 (维度: {self.unet.config.time_cond_proj_dim})")
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            print(f"   → 原始引导张量: shape={guidance_scale_tensor.shape} | dtype={guidance_scale_tensor.dtype}")
            
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, 
                embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
            print(f"✅ 条件嵌入生成完成: shape={timestep_cond.shape} | device={timestep_cond.device}")
        else:
            print("⚙️ 未配置时间条件投影")
            timestep_cond = None

        # 去噪循环初始化
        print("\n[阶段2] 循环参数配置")
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        print(f"📊 时间步总数: {len(timesteps)}")
        print(f"🔥 预热步数: {num_warmup_steps}")
        print(f"🔄 调度器顺序: {self.scheduler.order}阶")
        self._num_timesteps = len(timesteps)
        total_steps = len(timesteps)

        # 初始化性能监控
        # start_time = time.time()
        # step_times = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # step_start = time.time()
                current_step = i + 1
                
                print(f"\n[Step {current_step}/{total_steps}] 时间步: {t.item():.1f}")
                
                # 中断处理
                if self.interrupt:
                    print("⚠️ 检测到中断信号，跳过当前步骤")
                    continue

                # 潜在变量扩展
                print("\n[阶段2.1] 准备模型输入")
                if self.do_classifier_free_guidance:
                    print(f"🔀 扩展潜在变量 (引导比例: {self.guidance_scale})")
                    latent_model_input = torch.cat([latents] * 2)
                    print(f"   → 输入形状: {tuple(latent_model_input.shape)}")
                else:
                    print("⚙️ 直接使用原始潜在变量")
                    latent_model_input = latents
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                print(f"📐 缩放后输入范围: [{latent_model_input.min().item():.4f}, {latent_model_input.max().item():.4f}]")

                # UNet推理
                print("\n[阶段2.2] 噪声预测")
                print(f"🧠 UNet输入参数:")
                print(f"   → 时间步: {t.item():.1f}")
                print(f"   → 提示嵌入形状: {tuple(prompt_embeds.shape)}")
                if timestep_cond is not None:
                    print(f"   → 时间条件嵌入形状: {tuple(timestep_cond.shape)}")
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                print(f"✅ 噪声预测完成: shape={tuple(noise_pred.shape)}")

                # 引导处理
                print("\n[阶段2.3] 分类器自由引导")
                if self.do_classifier_free_guidance:
                    print(f"📊 分割噪声预测 (比例: {self.guidance_scale:.1f}x)")
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
                    noise_diff = noise_pred_text - noise_pred_uncond
                    print(f"   → 条件差异统计: μ={noise_diff.mean().item():.3f} ±{noise_diff.std().item():.3f}")
                    
                    noise_pred = noise_pred_uncond + self.guidance_scale * noise_diff
                    print(f"📈 引导后噪声范围: [{noise_pred.min().item():.3f}, {noise_pred.max().item():.3f}]")
                else:
                    print("⚙️ 跳过引导步骤")

            # def ok325322321():
                print("\n[Denoising Step] 开始单步去噪处理")
                
                # 引导重缩放处理
                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    print(f"\n🌀 应用噪声重缩放 (比例: {self.guidance_rescale:.2f})")
                    print(f"   → 参考算法: arXiv:2305.08891 第3.4节")
                    print(f"   → 原始噪声范围: [{noise_pred.min().item():.3f}, {noise_pred.max().item():.3f}]")
                    
                    noise_pred = rescale_noise_cfg(
                        noise_pred, 
                        noise_pred_text, 
                        guidance_rescale=self.guidance_rescale
                    )
                    print(f"✅ 重缩放后噪声统计:")
                    print(f"   → 均值变化: {noise_pred.mean().item()/noise_pred_text.mean().item():+.1%}")
                    print(f"   → 新值范围: [{noise_pred.min().item():.3f}, {noise_pred.max().item():.3f}]")
                else:
                    print("\n⚙️ 跳过噪声重缩放 (guidance_rescale={:.2f})".format(self.guidance_rescale))

                # 潜在变量更新
                print(f"\n⏳ 时间步 {t.item():.1f} 执行调度步骤...")
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                print(f"✅ 更新后潜在变量:")
                print(f"   → 形状: {latents.shape}")
                print(f"   → 均值: {latents.mean().item():.5f} (Δ{latents.mean().item() - noise_pred.mean().item():+.3e})")
                print(f"   → 标准差: {latents.std().item():.5f}")

                # 回调处理
                if callback_on_step_end is not None:
                    print("\n📡 执行步结束回调 ({}参数)".format(len(callback_on_step_end_tensor_inputs)))
                    callback_kwargs = {}
                    
                    # 构建回调参数
                    param_info = []
                    for k in callback_on_step_end_tensor_inputs:
                        val = locals().get(k, None)
                        callback_kwargs[k] = val
                        param_info.append(f"{k}: {type(val).__name__}{list(val.shape) if hasattr(val,'shape') else ''}")
                    print(f"   → 传递参数: {', '.join(param_info)}")
                    
                    # 执行回调
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    print(f"   → 回调返回 {len(callback_outputs)} 个修改项")
                    
                    # 应用修改
                    modified = []
                    for k in ["latents", "prompt_embeds", "negative_prompt_embeds"]:
                        if k in callback_outputs:
                            orig_shape = locals()[k].shape
                            locals()[k] = callback_outputs.pop(k)
                            modified.append(f"{k} {orig_shape} → {locals()[k].shape}")
                    if modified:
                        print(f"⚠️ 参数被修改: {' | '.join(modified)}")
                    else:
                        print("⚙️ 回调未修改核心参数")

                # 进度更新和回调触发
                update_flag = i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                )
                print(f"\n📊 进度更新条件: {'满足' if update_flag else '不满足'} [i={i}/步骤数={len(timesteps)}]")
                
                if update_flag:
                    prev_progress = progress_bar.n
                    progress_bar.update()
                    print(f"🔄 进度更新: {prev_progress} → {progress_bar.n}/{progress_bar.total}")
                    
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        print(f"📞 触发回调 (全局步 {step_idx})")
                        print(f"   → 当前时间步: {t.item():.1f}")
                        print(f"   → 潜在变量设备: {latents.device}")
                        callback(step_idx, t, latents)
                    else:
                        print(f"⏭️ 跳过回调 (步间隔 {callback_steps})")

                # XLA设备同步
                if XLA_AVAILABLE:
                    print("\n⚡ XLA设备同步")
                    print(f"   → 同步前内存: {xm.get_memory_info(xm.xla_device())['kb_free']/1024:.1f} MB 可用")
                    xm.mark_step()
                    print(f"   → 同步后设备状态: {xm.xla_device()}")
                    print(f"   → 同步后内存: {xm.get_memory_info(xm.xla_device())['kb_free']/1024:.1f} MB 可用")

    # print("\n[Denoising Step] 步骤处理完成 ✅\n")


    # def ok36557():
        print("\n[Postprocessing] 开始后处理流程")
        
        # VAE解码处理
        if not output_type == "latent":
            print(f"\n🔍 解码潜在变量 (缩放因子: {self.vae.config.scaling_factor})")
            print(f"   → 输入潜在变量形状: {latents.shape}")
            print(f"   → 数据类型: {latents.dtype}")
            
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, 
                return_dict=False, 
                generator=generator
            )[0]
            print(f"✅ 解码后图像形状: {image.shape} | 值域: [{image.min().item():.3f}, {image.max().item():.3f}]")

            # 安全检测
            print("\n🛡️ 执行内容安全检测...")
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            nsfw_count = sum(has_nsfw_concept) if has_nsfw_concept else 0
            print(f"   → 检测结果: 发现 {nsfw_count} 个NSFW内容" if nsfw_count > 0 
                else "   → 安全检测通过，未发现敏感内容")
        else:
            print("\n⚙️ 保持潜在变量输出")
            image = latents
            has_nsfw_concept = None
            print(f"   → 直接返回潜在变量形状: {image.shape}")

        # 反归一化处理
        print("\n🔧 准备反归一化参数")
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
            print(f"   → 全部 {image.shape[0]} 张图像将进行反归一化")
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            enabled = sum(do_denormalize)
            disabled = len(do_denormalize) - enabled
            print(f"   → 反归一化配置: 启用 {enabled} / 禁用 {disabled}")
        
        # 图像后处理
        print(f"\n🖼️ 执行最终图像处理 ({output_type.upper()})")
        image = self.image_processor.postprocess(
            image, 
            output_type=output_type, 
            do_denormalize=do_denormalize
        )
        print(f"✅ 处理后输出类型: {type(image[0]) if isinstance(image, list) else type(image)}")

        # 资源释放
        print("\n♻️ 释放模型资源")
        before_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.maybe_free_model_hooks()
        after_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            print(f"   → 显存释放: {(before_mem - after_mem)/1024**3:.2f} GB")

        # 返回结果处理
        print("\n📤 准备返回结果")
        if not return_dict:
            print(f"   → 返回元组格式 (图像, NSFW标记)")
            return (image, has_nsfw_concept)
        
        print("   → 返回结构化PipelineOutput")
        return StableDiffusionPipelineOutput(
            images=image, 
            nsfw_content_detected=has_nsfw_concept
        )

        print("\n[Postprocessing] 后处理完成 ✅\n")

