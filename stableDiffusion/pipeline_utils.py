# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import enum
import fnmatch
import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin

import numpy as np
import PIL.Image
import requests
import torch
from huggingface_hub import (
    ModelCard,
    create_repo,
    hf_hub_download,
    model_info,
    snapshot_download,
)
from huggingface_hub.utils import OfflineModeIsEnabled, validate_hf_hub_args
from packaging import version
from requests.exceptions import HTTPError
from tqdm.auto import tqdm

from diffusers import __version__
from configuration_utils import ConfigMixin
from diffusers.models import AutoencoderKL
from diffusers.models.attention_processor import FusedAttnProcessor2_0
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, ModelMixin
from diffusers.quantizers.bitsandbytes.utils import _check_bnb_status
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    BaseOutput,
    PushToHubMixin,
    is_accelerate_available,
    is_accelerate_version,
    is_torch_npu_available,
    is_torch_version,
    is_transformers_version,
    logging,
    numpy_to_pil,
)
from diffusers.utils.hub_utils import _check_legacy_sharding_variant_format, load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


if is_torch_npu_available():
    import torch_npu  # noqa: F401

from diffusers.pipelines.pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    CONNECTED_PIPES_KEYS,
    CUSTOM_PIPELINE_FILE_NAME,
    LOADABLE_CLASSES,
    _fetch_class_library_tuple,
    _get_custom_components_and_folders,
    _get_custom_pipeline_class,
    _get_final_device_map,
    _get_ignore_patterns,
    _get_pipeline_class,
    _identify_model_variants,
    _maybe_raise_warning_for_inpainting,
    _resolve_custom_pipeline_and_cls,
    _unwrap_model,
    _update_init_kwargs_with_connected_pipeline,
    load_sub_model,
    maybe_raise_or_warn,
    variant_compatible_siblings,
    warn_deprecated_model_variant,
)


if is_accelerate_available():
    import accelerate


LIBRARIES = []
for library in LOADABLE_CLASSES:
    LIBRARIES.append(library)

SUPPORTED_DEVICE_MAP = ["balanced"]

logger = logging.get_logger(__name__)


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised audio samples of a NumPy array of shape `(batch_size, num_channels, sample_rate)`.
    """

    audios: np.ndarray


class DiffusionPipeline(ConfigMixin, PushToHubMixin):
    r"""
    Base class for all pipelines.

    [`DiffusionPipeline`] stores all components (models, schedulers, and processors) for diffusion pipelines and
    provides methods for loading, downloading and saving models. It also includes methods to:

        - move all PyTorch modules to the device of your choice
        - enable/disable the progress bar for the denoising iteration

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.
        - **_optional_components** (`List[str]`) -- List of all optional components that don't have to be passed to the
          pipeline to function (should be overridden by subclasses).
    """

    config_name = "model_index.json"
    model_cpu_offload_seq = None
    hf_device_map = None
    _optional_components = []
    _exclude_from_cpu_offload = []
    _load_connected_pipes = False
    _is_onnx = False

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__ and hasattr(self.config, name):
            # We need to overwrite the config if name exists in config
            if isinstance(getattr(self.config, name), (tuple, list)):
                if value is not None and self.config[name][0] is not None:
                    class_library_tuple = _fetch_class_library_tuple(value)
                else:
                    class_library_tuple = (None, None)

                self.register_to_config(**{name: class_library_tuple})
            else:
                self.register_to_config(**{name: value})

        super().__setattr__(name, value)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Optional[Union[int, str]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~DiffusionPipeline.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a pipeline to. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            max_shard_size (`int` or `str`, defaults to `None`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5GB"`).
                If expressed as an integer, the unit is bytes. Note that this limit will be decreased after a certain
                period of time (starting from Oct 2024) to allow users to upgrade to the latest version of `diffusers`.
                This is to establish a common default size for this argument across different libraries in the Hugging
                Face ecosystem (`transformers`, and `accelerate`, for example).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            # Dynamo wraps the original model in a private class.
            # I didn't find a public API to get the original class.
            if is_compiled_module(sub_model):
                sub_model = _unwrap_model(sub_model)
                model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)
                else:
                    logger.info(
                        f"{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}"
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                logger.warning(
                    f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved."
                )
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters
            save_method_accept_max_shard_size = "max_shard_size" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant
            if save_method_accept_max_shard_size and max_shard_size is not None:
                # max_shard_size is expected to not be None in ModelMixin
                save_kwargs["max_shard_size"] = max_shard_size

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

        # finally save the config
        self.save_config(save_directory)

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token, is_pipeline=True)
            model_card = populate_model_card(model_card)
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    def to(self, *args, **kwargs):
        r"""
        Performs Pipeline dtype and/or device conversion. A torch.dtype and torch.device are inferred from the
        arguments of `self.to(*args, **kwargs).`

        <Tip>

            If the pipeline already has the correct torch.dtype and torch.device, then it is returned as is. Otherwise,
            the returned pipeline is a copy of self with the desired torch.dtype and torch.device.

        </Tip>


        Here are the ways to call `to`:

        - `to(dtype, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
        - `to(device, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        - `to(device=None, dtype=None, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the
          specified [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) and
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)

        Arguments:
            dtype (`torch.dtype`, *optional*):
                Returns a pipeline with the specified
                [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
            device (`torch.Device`, *optional*):
                Returns a pipeline with the specified
                [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
            silence_dtype_warnings (`str`, *optional*, defaults to `False`):
                Whether to omit warnings if the target `dtype` is not compatible with the target `device`.

        Returns:
            [`DiffusionPipeline`]: The pipeline converted to specified `dtype` and/or `dtype`.
        """
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        silence_dtype_warnings = kwargs.pop("silence_dtype_warnings", False)

        dtype_arg = None
        device_arg = None
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg
        pipeline_has_bnb = any(any((_check_bnb_status(module))) for _, module in self.components.items())

        # throw warning if pipeline is in "offloaded"-mode but user tries to manually set to GPU.
        def module_is_sequentially_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.14.0"):
                return False

            return hasattr(module, "_hf_hook") and (
                isinstance(module._hf_hook, accelerate.hooks.AlignDevicesHook)
                or hasattr(module._hf_hook, "hooks")
                and isinstance(module._hf_hook.hooks[0], accelerate.hooks.AlignDevicesHook)
            )

        def module_is_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.17.0.dev0"):
                return False

            return hasattr(module, "_hf_hook") and isinstance(module._hf_hook, accelerate.hooks.CpuOffload)

        # .to("cuda") would raise an error if the pipeline is sequentially offloaded, so we raise our own to make it clearer
        pipeline_is_sequentially_offloaded = any(
            module_is_sequentially_offloaded(module) for _, module in self.components.items()
        )
        if device and torch.device(device).type == "cuda":
            if pipeline_is_sequentially_offloaded and not pipeline_has_bnb:
                raise ValueError(
                    "It seems like you have activated sequential model offloading by calling `enable_sequential_cpu_offload`, but are now attempting to move the pipeline to GPU. This is not compatible with offloading. Please, move your pipeline `.to('cpu')` or consider removing the move altogether if you use sequential offloading."
                )
            # PR: https://github.com/huggingface/accelerate/pull/3223/
            elif pipeline_has_bnb and is_accelerate_version("<", "1.1.0.dev0"):
                raise ValueError(
                    "You are trying to call `.to('cuda')` on a pipeline that has models quantized with `bitsandbytes`. Your current `accelerate` installation does not support it. Please upgrade the installation."
                )

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline which doesn't allow explicit device placement using `to()`. You can call `reset_device_map()` first and then call `to()`."
            )

        # Display a warning in this case (the operation succeeds but the benefits are lost)
        pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in self.components.items())
        if pipeline_is_offloaded and device and torch.device(device).type == "cuda":
            logger.warning(
                f"It seems like you have activated model offloading by calling `enable_model_cpu_offload`, but are now manually moving the pipeline to GPU. It is strongly recommended against doing so as memory gains from offloading are likely to be lost. Offloading automatically takes care of moving the individual components {', '.join(self.components.keys())} to GPU when needed. To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading."
            )

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
        for module in modules:
            _, is_loaded_in_4bit_bnb, is_loaded_in_8bit_bnb = _check_bnb_status(module)

            if (is_loaded_in_4bit_bnb or is_loaded_in_8bit_bnb) and dtype is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in `bitsandbytes` {'4bit' if is_loaded_in_4bit_bnb else '8bit'} and conversion to {dtype} is not supported. Module is still in {'4bit' if is_loaded_in_4bit_bnb else '8bit'} precision."
                )

            if is_loaded_in_8bit_bnb and device is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in `bitsandbytes` 8bit and moving it to {device} via `.to()` is not supported. Module is still on {module.device}."
                )

            # This can happen for `transformer` models. CPU placement was added in
            # https://github.com/huggingface/transformers/pull/33122. So, we guard this accordingly.
            if is_loaded_in_4bit_bnb and device is not None and is_transformers_version(">", "4.44.0"):
                module.to(device=device)
            elif not is_loaded_in_4bit_bnb and not is_loaded_in_8bit_bnb:
                module.to(device, dtype)

            if (
                module.dtype == torch.float16
                and str(device) in ["cpu"]
                and not silence_dtype_warnings
                and not is_offloaded
            ):
                logger.warning(
                    "Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device. It"
                    " is not recommended to move them to `cpu` as running them will fail. Please make"
                    " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                    " support for`float16` operations on this device in PyTorch. Please, remove the"
                    " `torch_dtype=torch.float16` argument, or use another device for inference."
                )
        return self

    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.device

        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns:
            `torch.dtype`: The torch dtype on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.dtype

        return torch.float32

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers
                version.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
        # Copy the kwargs to re-use during loading connected pipeline.
        kwargs_copied = kwargs.copy()
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)

    # def ok32432():
        print("\n[Config Initialization] 开始配置初始化流程")
        print("🌀 正在复制初始参数表...")
        kwargs_copied = kwargs.copy()
        print(f"✅ 参数表备份完成 | 原始参数数量: {len(kwargs)} 个")

        # 参数提取表格
        param_table = [
            ("cache_dir", "模型缓存路径", None),
            ("force_download", "强制下载", False),
            ("proxies", "网络代理", None),
            ("local_files_only", "仅本地文件", None),
            ("token", "访问令牌", None),
            ("revision", "代码版本", None),
            ("from_flax", "Flax模型", False),
            ("torch_dtype", "张量精度", None),
            ("custom_pipeline", "自定义管道", None),
            ("custom_revision", "自定义版本", None),
            ("provider", "ONNX提供商", None),
            ("sess_options", "会话配置", None),
            ("device_map", "设备映射", None),
            ("max_memory", "显存分配", None),
            ("offload_folder", "卸载目录", None),
            ("offload_state_dict", "状态字典卸载", False),
            ("low_cpu_mem_usage", "低内存模式", _LOW_CPU_MEM_USAGE_DEFAULT),
            ("variant", "变体名称", None),
            ("use_safetensors", "安全张量格式", None),
            ("use_onnx", "ONNX运行时", None),
            ("load_connected_pipeline", "连接管道加载", False)
        ]

        print("\n[阶段1] 提取运行时参数")
        extracted_params = {}
        for param, desc, default in param_table:
            value = kwargs.pop(param, default)
            status = "使用默认值" if value == default else "自定义配置"
            extracted_params[param] = value
            print(f"   → {param.ljust(20)}: {str(value).ljust(15)} | {status} ({desc})")

        # 特殊参数处理显示
        print("\n🔧 特殊参数解析：")
        print(f"   → from_flax 转换状态: {'是' if from_flax else '否'}")
        print(f"   → torch_dtype 精度设置: {torch_dtype if torch_dtype else '自动检测'}")
        print(f"   → 设备映射策略: {device_map if device_map else '自动分配'}")

        # 剩余参数警告
        print("\n⚠️ 未处理参数检查:")
        if kwargs:
            print(f"发现 {len(kwargs)} 个未识别参数:")
            for key in list(kwargs.keys())[:3]:  # 最多显示前3个
                print(f"   → {key}: {type(kwargs[key]).__name__}")
            if len(kwargs) > 3:
                print(f"   ...（其余 {len(kwargs)-3} 个参数未显示）")
        else:
            print("✅ 所有参数已正确解析")

        print("\n[阶段2] 连接管道加载准备")
        print(f"🔗 连接管道加载状态: {'已启用' if load_connected_pipeline else '已禁用'}")
        if load_connected_pipeline:
            print("   → 正在保留原始参数表用于管道连接")
            print(f"   → 保留参数数量: {len(kwargs_copied)} 个")
        else:
            print("   → 标准加载模式，无需额外处理")

        print("\n[Config Initialization] 配置初始化完成 ✅")
        print(f"最终提取参数: {len(extracted_params)} 项 | 剩余参数: {len(kwargs)} 项\n")


    # def ok43242():
        print("\n[System Validation] 开始系统验证流程")
        
        # 低内存模式检测
        print("\n[阶段1] 验证低内存配置")
        if low_cpu_mem_usage and not is_accelerate_available():
            print("⚠️ 检测到环境不兼容：")
            print("   → 请求低内存模式: True")
            print("   → accelerate 包状态: 未安装")
            
            low_cpu_mem_usage = False
            print("🔄 已自动禁用低内存模式 | low_cpu_mem_usage=False")
            
            warning_msg = (
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )
            print(f"📢 重要警告：{warning_msg[:120]}...")

        # PyTorch版本检测（低内存模式）
        print("\n[阶段2] 检查PyTorch版本兼容性")
        if low_cpu_mem_usage is True:
            print("🔍 验证低内存模式要求：")
            torch_version = torch.__version__
            required_version = "1.9.0"
            status = "符合" if is_torch_version(">=", required_version) else "不符合"
            
            print(f"   → 当前PyTorch版本: {torch_version}")
            print(f"   → 最低要求版本: {required_version} | 状态: {status}")
            
            if not is_torch_version(">=", "1.9.0"):
                error_msg = (
                    "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                    " `low_cpu_mem_usage=False`."
                )
                print(f"❌ 版本不兼容错误：{error_msg}")
                raise NotImplementedError(error_msg)
        else:
            print("✅ 低内存模式未启用，跳过版本检查")

        # 设备映射验证流程
        print("\n[阶段3] 验证设备映射配置")
        if device_map is not None:
            print(f"🔧 设备映射配置检测：{device_map}")
            
            # 版本检查
            print("   → 检查PyTorch版本要求...")
            torch_version = torch.__version__
            required_version = "1.9.0"
            if not is_torch_version(">=", required_version):
                error_msg = (
                    f"Loading and dispatching requires torch >= {required_version} (当前版本: {torch_version}). "
                    "Please either update your PyTorch version or set `device_map=None`."
                )
                print(f"❌ 版本错误：{error_msg}")
                raise NotImplementedError(error_msg)
            
            # 加速库检查
            print("   → 检查accelerate依赖...")
            if not is_accelerate_available():
                error_msg = "Using `device_map` requires the `accelerate` library. Please install using: `pip install accelerate`."
                print(f"❌ 依赖缺失：{error_msg}")
                raise NotImplementedError(error_msg)
            
            # 类型检查
            print("   → 验证配置类型...")
            if not isinstance(device_map, str):
                error_msg = f"设备映射必须为字符串类型，检测到类型：{type(device_map).__name__}"
                print(f"❌ 类型错误：{error_msg}")
                raise ValueError("`device_map` must be a string.")
            
            # 策略支持检查
            print("   → 验证策略兼容性...")
            print(f"      支持策略列表: {', '.join(SUPPORTED_DEVICE_MAP)}")
            if device_map not in SUPPORTED_DEVICE_MAP:
                error_msg = (
                    f"不支持的设备映射策略 '{device_map}'，可用策略：{', '.join(SUPPORTED_DEVICE_MAP)}"
                )
                print(f"❌ 策略错误：{error_msg}")
                raise NotImplementedError(error_msg)


    # def ok32432():
        print("\n[Model Loading] 开始模型加载流程")
        
        # 设备映射与加速库版本检查
        print("\n[阶段1] 验证设备映射兼容性")
        if device_map is not None and device_map in SUPPORTED_DEVICE_MAP:
            print(f"🔧 检测到设备映射策略: {device_map}")
            print("   → 检查加速库版本要求...")
            required_accelerate = "0.28.0"
            current_accelerate = get_accelerate_version() if is_accelerate_available() else "未安装"
            
            print(f"   → 当前accelerate版本: {current_accelerate}")
            print(f"   → 需要的最低版本: {required_accelerate}")
            
            if is_accelerate_version("<", required_accelerate):
                error_msg = f"设备映射需要accelerate>={required_accelerate}，当前版本：{current_accelerate}"
                print(f"❌ 版本不兼容: {error_msg}")
                raise NotImplementedError("Device placement requires `accelerate` version `0.28.0` or later.")
            print("✅ 加速库版本验证通过")

        # 参数冲突检测
        print("\n[阶段2] 检查参数兼容性")
        if low_cpu_mem_usage is False and device_map is not None:
            print("⚠️ 检测到参数冲突：")
            print(f"   → low_cpu_mem_usage={low_cpu_mem_usage}")
            print(f"   → device_map={device_map}")
            error_msg = (
                f"不能同时设置low_cpu_mem_usage=False和使用device_map={device_map}，"
                "请设置low_cpu_mem_usage=True"
            )
            print(f"❌ 参数冲突错误: {error_msg}")
            raise ValueError(error_msg)
        else:
            print("✅ 参数组合验证通过")

        # 模型路径处理
        print("\n[阶段3] 处理模型路径")
        print(f"📂 原始模型路径: {pretrained_model_name_or_path}")
        
        if not os.path.isdir(pretrained_model_name_or_path):
            print("🔄 检测到远程模型标识符")
            if pretrained_model_name_or_path.count("/") > 1:
                print(f"⚠️ 路径格式错误: 包含{pretrained_model_name_or_path.count('/')}个分隔符")
                error_msg = f'无效路径或仓库ID: "{pretrained_model_name_or_path}"'
                print(f"❌ 路径验证失败: {error_msg}")
                raise ValueError(error_msg)
            
            # 模型下载流程
            print("\n[阶段4] 开始模型下载")
            print("🔍 下载参数摘要:")
            print(f"   → 缓存目录: {cache_dir or '默认'}")
            print(f"   → 强制下载: {force_download}")
            print(f"   → 代理设置: {proxies.keys() if proxies else '无'}")
            print(f"   → 安全张量: {use_safetensors}")
            
            print("⏳ 正在下载模型文件...")
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                from_flax=from_flax,
                use_safetensors=use_safetensors,
                use_onnx=use_onnx,
                custom_pipeline=custom_pipeline,
                custom_revision=custom_revision,
                variant=variant,
                load_connected_pipeline=load_connected_pipeline,
                **kwargs,
            )
            print(f"✅ 下载完成 | 缓存路径: {cached_folder}")
        else:
            print("🔍 使用本地模型目录")
            cached_folder = pretrained_model_name_or_path
            print(f"📂 本地模型路径: {cached_folder}")

    # def ok23432():
        print("\n[Config Processing] 开始配置处理流程")
        
        # 分片格式检查
        print("\n[阶段1] 验证分片文件格式")
        if variant is not None:
            print(f"🔍 正在检查变体 '{variant}' 的分片格式...")
            legacy_check_result = _check_legacy_sharding_variant_format(
                folder=cached_folder, 
                variant=variant
            )
            
            if legacy_check_result:
                print(f"⚠️ 检测到旧版分片格式 | 路径: {cached_folder}")
                warn_msg = (
                    f"Warning: The repository contains sharded checkpoints for variant '{variant}' maybe in a deprecated format. "
                    "Please check your files carefully:\n\n"
                    "- Correct format example: diffusion_pytorch_model.fp16-00003-of-00003.safetensors\n"
                    "- Deprecated format example: diffusion_pytorch_model-00001-of-00002.fp16.safetensors\n\n"
                    "If you find any files in the deprecated format:\n"
                    "1. Remove all existing checkpoint files for this variant.\n"
                    "2. Re-obtain the correct files by running `save_pretrained()`.\n\n"
                    "This will ensure you're using the most up-to-date and compatible checkpoint format."
                )
                print(f"▌{' 格式警告 '.center(50, '─')}")
                for line in warn_msg.split('\n')[:4]:  # 显示前4行关键信息
                    print(f"│ {line}")
                print(f"└{'─'*50}")
            else:
                print(f"✅ 分片格式验证通过 | 变体: {variant}")
        else:
            print("⚙️ 未指定变体名称，跳过分片格式检查")

        # 配置文件加载
        print("\n[阶段2] 加载模型配置", "用到的类：", cls)
        print(f"📂 从缓存目录加载配置: {cached_folder}")
        config_dict = cls.load_config(cached_folder)
        print(f"✅ 配置加载完成 | 共加载 {len(config_dict)} 个配置项")
        
        # 清理临时配置项
        print("\n[阶段3] 清理配置数据")
        ignored_keys = config_dict.pop("_ignore_files", None)
        if ignored_keys:
            print(f"🗑️ 移除临时配置项 '_ignore_files' | 包含 {len(ignored_keys)} 个忽略规则")
            print(f"   → 示例规则: {ignored_keys[0]}") if ignored_keys else None
        else:
            print("✅ 无需清理，未找到 '_ignore_files' 配置项")

        # print("\n[Config Processing] 配置处理完成 ✅\n")


        # 2. Define which model components should load variants
        # We retrieve the information by matching whether variant model checkpoints exist in the subfolders.
        # Example: `diffusion_pytorch_model.safetensors` -> `diffusion_pytorch_model.fp16.safetensors`
        # with variant being `"fp16"`.
    # def ok32432():
        print("\n[Model Initialization] 开始模型初始化流程")
        
        # 模型变体验证
        print("\n[阶段1] 验证模型变体配置")
        print(f"🔍 正在识别变体文件 | 指定变体: {variant or '未指定'}")
        model_variants = _identify_model_variants(
            folder=cached_folder, 
            variant=variant, 
            config=config_dict
        )
        
        if len(model_variants) == 0 and variant is not None:
            error_message = (
                f"无法加载指定变体 '{variant}' 的模型文件，"
                f"在路径 {cached_folder} 中未找到匹配文件"
            )
            print(f"❌ 变体验证失败: {error_message}")
            raise ValueError(error_message)
        elif variant is not None:
            print(f"✅ 变体验证通过 | 找到 {len(model_variants)} 个变体文件")
            print(f"   → 示例文件: {model_variants[0]}" if model_variants else "")
        else:
            print("⚙️ 未指定变体参数，跳过变体检查")

        # 自定义管道解析
        print("\n[阶段2] 解析自定义管道配置")
        print(f"🔧 输入参数 custom_pipeline: {custom_pipeline or '未指定'}")
        custom_pipeline, custom_class_name = _resolve_custom_pipeline_and_cls(
            folder=cached_folder, 
            config=config_dict, 
            custom_pipeline=custom_pipeline
        )
        
        if custom_pipeline:
            source = "用户指定" if custom_pipeline else "配置自动检测"
            print(f"🔄 使用自定义管道 | 来源: {source}")
            print(f"   → 管道名称: {custom_pipeline}")
            print(f"   → 类名称: {custom_class_name}")
        else:
            print("✅ 使用标准管道配置")

        # 获取管道类
        print("\n[阶段3] 加载管道类别")
        pipeline_class = _get_pipeline_class(
            cls,
            config=config_dict,
            load_connected_pipeline=load_connected_pipeline,
            custom_pipeline=custom_pipeline,
            class_name=custom_class_name,
            cache_dir=cache_dir,
            revision=custom_revision,
        )
        
        print(f"🏭 管道类已加载 | 类型: {pipeline_class.__name__}")
        print(f"   → 来源: {'自定义实现' if custom_pipeline else '标准实现'}")
        print(f"   → 修订版本: {custom_revision or '默认'}")
        print(f"   → 缓存目录: {cache_dir or '系统默认'}")

        # 设备映射兼容性检查
        print("\n[阶段4] 验证设备映射兼容性")
        if device_map is not None and pipeline_class._load_connected_pipes:
            error_msg = (
                "当前设备映射配置与连接管道不兼容\n"
                f"   → 设备映射: {device_map}\n"
                f"   → 连接管道: {pipeline_class._load_connected_pipes}\n"
                "解决方案建议:\n"
                "1. 禁用设备映射 (设置 device_map=None)\n"
                "2. 使用独立管道替代连接管道"
            )
            print(f"❌ 兼容性错误: {error_msg}")
            raise NotImplementedError("`device_map` is not yet supported for connected pipelines.")
        else:
            print("✅ 设备映射验证通过")



        # DEPRECATED: To be removed in 1.0.0
        # we are deprecating the `StableDiffusionInpaintPipelineLegacy` pipeline which gets loaded
        # when a user requests for a `StableDiffusionInpaintPipeline` with `diffusers` version being <= 0.5.1.
    # def ok324324():
        print("\n[Inpainting Check] 开始图像修复兼容性检查")
        
        print("🔍 正在验证模型修复能力兼容性:")
        print(f"   → 管道类型: {pipeline_class.__name__}")
        print(f"   → 模型路径: {pretrained_model_name_or_path}")
        
        # 执行兼容性检查
        print("\n[阶段1] 分析模型配置")
        if not hasattr(config_dict, "get"):
            print("⚠️ 配置字典异常: 缺少标准配置方法")
            return
        
        print("   → 检查修复相关参数...")
        requires_inpainting = config_dict.get("requires_inpainting", False)
        print(f"      修复功能需求标识: {requires_inpainting}")
        
        print("\n[阶段2] 执行版本兼容性检测")
        warning_triggered = _maybe_raise_warning_for_inpainting(
            pipeline_class=pipeline_class,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config_dict,
        )
        
        if warning_triggered:
            print("\n⚠️ 图像修复警告已触发")
            print("▌" + " 重要提示 ".center(50, '─'))
            print("│ 检测到可能不兼容的修复模型配置")
            print("│ 建议操作:")
            print("│ 1. 确认模型是否专门用于修复任务")
            print("│ 2. 检查模型配置文件中的参数")
            print("│ 3. 使用官方推荐的修复模型版本")
            print(f"└{'─'*50}")
        else:
            print("✅ 图像修复兼容性检查通过")
        
        # print("\n[Inpainting Check] 检查流程完成\n")


        # 4. Define expected modules given pipeline signature
        # and define non-None initialized modules (=`init_kwargs`)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
    
    # def ok23432():
        print("\n[Parameter Processing] 开始参数处理流程")
        
        # 获取签名参数
        print("\n[阶段1] 解析管道签名参数")
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        print(f"📝 预期模块参数 ({len(expected_modules)}个):")
        print("   → " + ", ".join(expected_modules))
        print(f"🔧 可选关键字参数 ({len(optional_kwargs)}个):")
        print("   → " + ", ".join(optional_kwargs))

        # 获取类型签名
        print("\n[阶段2] 获取类型约束")
        expected_types = pipeline_class._get_signature_types()
        print(f"🔍 检测到 {len(expected_types)} 个类型约束:")
        # for param, dtype in list(expected_types.items())[:3]:  # 显示前3个示例
        #     print(f"   → {param}: {dtype.__name__}")
        # if len(expected_types) > 3:
        #     print(f"   → ...(其余{len(expected_types)-3}个类型约束省略)")

        # 参数分类处理
        print("\n[阶段3] 分类输入参数")
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        print(f"🎯 显式传递的类对象参数 ({len(passed_class_obj)}个):")
        for k, v in passed_class_obj.items():
            print(f"   → {k}: {type(v).__name__}")

        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        print(f"\n⚙️ 管道专用关键字参数 ({len(passed_pipe_kwargs)}个):")
        for k, v in passed_pipe_kwargs.items():
            print(f"   → {k}: {v}")

        # 提取初始化字典
        print("\n[阶段4] 提取初始化配置")
        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)
        print(f"📦 基础初始化字典 ({len(init_dict)}项):")
        print("   → 示例键: " + ", ".join(list(init_dict.keys())[:3]))
        print(f"\n🚫 未使用参数 ({len(unused_kwargs)}个):")
        print("   → " + ", ".join(unused_kwargs.keys()))

        # 构建初始化关键字参数
        print("\n[阶段5] 构建最终初始化参数")
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        print(f"🔨 合并必要参数 ({len(init_kwargs)}项):")
        print("   → 包含参数: " + ", ".join(init_kwargs.keys()))
        
        print("\n🔄 合并管道专用参数")
        init_kwargs = {**init_kwargs,**passed_pipe_kwargs}
        print(f"✅ 最终初始化参数 ({len(init_kwargs)}项):")
        for k, v in list(init_kwargs.items())[:3]:
            print(f"   → {k}: {v if not callable(v) else 'callable'}")

        # 过滤空组件
        print("\n[阶段6] 过滤无效组件")
        def load_module(name, value):
            print(f"   ├─ 检查组件: {name}")
            if value[0] is None:
                print(f"   │  → 排除原因: 配置值为None")
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                print(f"   │  → 排除原因: 显式传递为None")
                return False
            print(f"   └─ 保留组件: {name}")
            return True

        print("🔍 开始组件过滤...")
        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}
        print(f"\n✅ 过滤后有效组件 ({len(init_dict)}个):")
        print("   → " + ", ".join(init_dict.keys()))

        print("\n[Parameter Processing] 参数处理完成 ✅\n")

    # def ok32432():
        print("\n[Component Validation] 开始组件验证流程")
        
        # 组件类型校验
        print("\n[阶段1] 执行组件类型检查")
        for key in init_dict.keys():
            print(f"\n🔍 检查组件 '{key}'...")
            
            if key not in passed_class_obj:
                print(f"   → 未在显式传递参数中，跳过校验")
                continue
                
            if "scheduler" in key:
                print(f"   → 调度器组件跳过类型检查")
                continue
                
            class_obj = passed_class_obj[key]
            print(f"   → 对象类型: {class_obj.__class__.__name__}")
            
            # 生成预期类型列表
            _expected_class_types = []
            for expected_type in expected_types[key]:
                if isinstance(expected_type, enum.EnumMeta):
                    print(f"   → 解析枚举类型: {expected_type.__name__}")
                    _expected_class_types.extend(expected_type.__members__.keys())
                else:
                    _expected_class_types.append(expected_type.__name__)
                    
            print(f"   → 允许的类型列表: {', '.join(_expected_class_types)}")
            
            # 执行类型验证
            _is_valid_type = class_obj.__class__.__name__ in _expected_class_types
            if _is_valid_type:
                print(f"✅ 类型校验通过")
            else:
                print(f"❌ 类型不匹配!")
                logger.warning(
                    f"Expected types for {key}: {_expected_class_types}, got {class_obj.__class__.__name__}."
                )

        # Flax特殊检测
        print("\n[阶段2] 验证Flax加载模式")
        if from_flax and "safety_checker" in init_dict and "safety_checker" not in passed_class_obj:
            print("⚠️ 检测到Flax加载的安全隐患:")
            print("   → from_flax = True")
            print("   → safety_checker存在但未显式传递")
            print("   → 解决方案提示:")
            print("     1. 设置 safety_checker=None")
            print("     2. 手动加载安全检查器")
            raise NotImplementedError(
                "The safety checker cannot be automatically loaded when loading weights `from_flax`."
                " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
                " separately if you need it."
            )
        else:
            print("✅ Flax加载安全检查通过")

        # 未使用参数警告
        print("\n[阶段3] 分析未使用参数")
        if len(unused_kwargs) > 0:
            print(f"⚠️ 发现{len(unused_kwargs)}个未识别参数:")
            for i, (k, v) in enumerate(unused_kwargs.items()):
                if i < 3:  # 最多显示前3个
                    print(f"   → {k}: {v.__class__.__name__ if not isinstance(v, str) else v}")
            if len(unused_kwargs) > 3:
                print(f"   → ...(其余{len(unused_kwargs)-3}个参数省略)")
            logger.warning(
                f"Keyword arguments {list(unused_kwargs.keys())} are not expected by {pipeline_class.__name__} and will be ignored."
            )
        else:
            print("✅ 所有参数均被正确使用")


        # import it here to avoid circular import
    # def ok23432():
        from diffusers import pipelines

    # def ok23432():
        print("\n[Device Mapping] 开始设备映射处理")
        
        # 设备映射初始化
        final_device_map = None
        if device_map is not None:
            print(f"\n🔧 初始化设备映射 | 策略: {device_map}")
            print("▌" + " 设备映射参数 ".center(50, '─'))
            print(f"│ → 最大显存分配: {max_memory or '默认'}")
            print(f"│ → 张量精度: {torch_dtype}")
            print(f"│ → 缓存路径: {cached_folder}")
            print(f"└{'─'*50}")
            
            final_device_map = _get_final_device_map(
                device_map=device_map,
                pipeline_class=pipeline_class,
                passed_class_obj=passed_class_obj,
                init_dict=init_dict,
                library=library,
                max_memory=max_memory,
                torch_dtype=torch_dtype,
                cached_folder=cached_folder,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
            print(f"\n✅ 最终设备映射生成完成 | 包含 {len(final_device_map)} 个组件分配")
            print(f"   → 示例分配: {dict(list(final_device_map.items())[:2])}...")
        else:
            print("⚙️ 未配置设备映射，跳过分配")

        print("\n[Module Loading] 开始模块加载流程")
        current_device_map = None
        
        # 模块加载进度
        total_modules = len(init_dict.items())
        print(f"📦 共需加载 {total_modules} 个组件")
        
        for idx, (name, (library_name, class_name)) in enumerate(init_dict.items(), 1):
            print(f"\n🔌 正在加载组件 ({idx}/{total_modules}): {name}")
            print(f"   → 所属库: {library_name}")
            print(f"   → 原始类名: {class_name}")

            # 设备映射处理
            if final_device_map:
                component_device = final_device_map.get(name, None)
                print(f"   → 分配设备: {component_device or '自动分配'}")
                current_device_map = {"": component_device} if component_device else None
            else:
                print("   → 设备策略: 全局默认")

            # Flax类名处理
            if class_name.startswith("Flax"):
                original_class = class_name
                class_name = class_name[4:]
                print(f"   🪓 调整Flax类名: {original_class} → {class_name}")

            # 导入类检查
            is_pipeline_module = hasattr(pipelines, library_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            print(f"   → 是否管道模块: {'是' if is_pipeline_module else '否'}")
            
            # 已传递对象检查
            if name in passed_class_obj:
                print(f"\n⚠️ 使用预传递的 {name} 组件")
                print(f"   → 对象类型: {type(passed_class_obj[name]).__name__}")
                
                # 执行父类校验
                print("   → 执行父类合规性检查...")
                maybe_raise_or_warn(
                    library_name, library, class_name, 
                    ALL_IMPORTABLE_CLASSES, passed_class_obj, name, is_pipeline_module
                )
                loaded_sub_model = passed_class_obj[name]
                print(f"✅ 预传递组件验证通过 | 设备: {get_device(loaded_sub_model)}")
            else:
            # def ok324324():
                print("\n[Submodel Loading] 开始子模型加载流程")
                
                # 加载子模型
                print(f"🔧 加载参数摘要:")
                print(f"   → 所属库: {library_name}")
                print(f"   → 目标类名: {class_name}")
                print(f"   → 设备映射策略: {current_device_map or '默认分配'}")
                print(f"   → 张量精度: {torch_dtype or '自动检测'}")
                print(f"   → 使用安全张量: {'是' if use_safetensors else '否'}")
                if variant:
                    print(f"   → 使用变体配置: {variant}")
                if class_name == "UNet2DConditionModel":
                    library_name = "unet_2d_condition"
                # try:
                print("\n⏳ 正在加载子模型...")
                loaded_sub_model = load_sub_model(
                    library_name=library_name,
                    class_name=class_name,
                    importable_classes=importable_classes,
                    pipelines=pipelines,
                    is_pipeline_module=is_pipeline_module,
                    pipeline_class=pipeline_class,
                    torch_dtype=torch_dtype,
                    provider=provider,
                    sess_options=sess_options,
                    device_map=current_device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    model_variants=model_variants,
                    name=name,
                    from_flax=from_flax,
                    variant=variant,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    cached_folder=cached_folder,
                    use_safetensors=use_safetensors,
                )
                
                # 显示加载结果
                # print(f"\n✅ 子模型加载成功 | 设备位置: {get_device(loaded_sub_model)}")
                # print(f"   → 实际数据类型: {loaded_sub_model.dtype}")
                # print(f"   → 内存占用: {get_model_memory_usage(loaded_sub_model):.2f}MB")
                if hasattr(loaded_sub_model, "num_parameters"):
                    print(f"   → 参数量: {loaded_sub_model.num_parameters()/1e6:.1f}M")
                    
                # 特殊模式提示
                if from_flax:
                    print("⚠️ 注意: 当前使用Flax框架加载权重")
                if low_cpu_mem_usage:
                    print("💾 低内存模式已启用")
                    
                logger.info(
                    f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
                )           

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

    # def ok32432432():
        print("\n[Final Initialization] 开始最终初始化流程")
        
        # 处理连接管道
        print("\n[阶段1] 处理连接管道配置")
        if pipeline_class._load_connected_pipes:
            readme_path = os.path.join(cached_folder, "README.md")
            print(f"🔗 检查连接管道需求 | README存在: {os.path.isfile(readme_path)}")
            
            if os.path.isfile(readme_path):
                print("🔄 检测到连接管道配置，更新初始化参数...")
                init_kwargs_before = set(init_kwargs.keys())
                
                init_kwargs = _update_init_kwargs_with_connected_pipeline(
                    init_kwargs=init_kwargs,
                    passed_pipe_kwargs=passed_pipe_kwargs,
                    passed_class_objs=passed_class_obj,
                    folder=cached_folder,
                    **kwargs_copied,
                )
                
                added_params = set(init_kwargs.keys()) - init_kwargs_before
                print(f"✅ 新增连接管道参数 ({len(added_params)}项):")
                print("   → " + ", ".join(added_params))
        else:
            print("⚙️ 未启用连接管道加载功能")

        # 缺失模块处理
        print("\n[阶段2] 验证模块完整性")
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        print(f"🔍 缺失必要模块 ({len(missing_modules)}个):")
        print("   → " + (", ".join(missing_modules) if missing_modules else "无"))
        
        optional_modules = pipeline_class._optional_components
        print(f"📦 可选模块清单 ({len(optional_modules)}个):")
        print("   → " + ", ".join(optional_modules))
        
        if missing_modules:
            print("\n⚖️ 开始模块兼容性评估...")
            covered_modules = missing_modules <= set(passed_class_obj.keys() | optional_modules)
            print(f"   → 缺失模块是否可覆盖: {'是' if covered_modules else '否'}")
            
            if covered_modules:
                print("🛠️ 自动补全缺失模块:")
                for module in missing_modules:
                    source = "显式传递" if module in passed_class_obj else "设为None"
                    init_kwargs[module] = passed_class_obj.get(module, None)
                    print(f"   → {module.ljust(15)}: {source}")
            else:
                print("❌ 存在无法覆盖的缺失模块")
                passed_modules = set(init_kwargs.keys()) | set(passed_class_obj.keys()) - set(optional_modules)
                error_msg = (
                    f"模块完整性检查失败\n"
                    f"预期模块: {', '.join(expected_modules)}\n"
                    f"已提供模块: {', '.join(passed_modules)}\n"
                    f"缺失关键模块: {', '.join(missing_modules - optional_modules)}"
                )
                print(error_msg)
                raise ValueError(error_msg)
        else:
            print("✅ 所有必要模块已就绪")

        # 实例化管道
        print("\n[阶段3] 创建管道实例")
        print(f"🏭 初始化参数摘要 ({len(init_kwargs)}项):")
        for k, v in list(init_kwargs.items())[:3]:  # 显示前3个参数示例
            print(f"   → {k.ljust(15)}: {type(v).__name__ if v else 'None'}")
        if len(init_kwargs) > 3:
            print(f"   → ...(其余{len(init_kwargs)-3}个参数省略)")
        
        model = pipeline_class(**init_kwargs)
        print(f"✅ 管道实例创建成功 | 类型: {model.__class__.__name__}")

        # 保存配置信息
        print("\n[阶段4] 注册配置信息")
        print(f"📝 记录模型来源路径: {pretrained_model_name_or_path}")
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        
        if device_map is not None:
            print(f"📌 记录设备映射策略:")
            for device, components in final_device_map.items():
                print(f"   → {device.ljust(8)}: {len(components)}个组件")
            setattr(model, "hf_device_map", final_device_map)
        else:
            print("⚙️ 未配置设备映射，跳过记录")

        print("\n[Final Initialization] 初始化流程全部完成 ✅\n")

        return model

    @property
    def name_or_path(self) -> str:
        return getattr(self.config, "_name_or_path", None)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def remove_all_hooks(self):
        r"""
        Removes all hooks that were added when using `enable_sequential_cpu_offload` or `enable_model_cpu_offload`.
        """
        for _, model in self.components.items():
            if isinstance(model, torch.nn.Module) and hasattr(model, "_hf_hook"):
                accelerate.hooks.remove_hook_from_module(model, recurse=True)
        self._all_hooks = []

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_model_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_model_cpu_offload()`."
            )

        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        self.remove_all_hooks()

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        self.to("cpu", silence_dtype_warnings=True)
        device_mod = getattr(torch, device.type, None)
        if hasattr(device_mod, "empty_cache") and device_mod.is_available():
            device_mod.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)

            if not isinstance(model, torch.nn.Module):
                continue

            # This is because the model would already be placed on a CUDA device.
            _, _, is_loaded_in_8bit_bnb = _check_bnb_status(model)
            if is_loaded_in_8bit_bnb:
                logger.info(
                    f"Skipping the hook placement for the {model.__class__.__name__} as it is loaded in `bitsandbytes` 8bit."
                )
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload(device=getattr(self, "_offload_device", "cuda"))

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using 🤗 Accelerate, significantly reducing memory usage. When called, the state
        dicts of all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are saved to CPU
        and then moved to `torch.device('meta')` and loaded to GPU only when their specific submodule has its `forward`
        method called. Offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")
        self.remove_all_hooks()

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_sequential_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_sequential_cpu_offload()`."
            )

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                # make sure to offload buffers if not all high level weights
                # are of type nn.Module
                offload_buffers = len(model._parameters) > 0
                cpu_offload(model, device, offload_buffers=offload_buffers)

    def reset_device_map(self):
        r"""
        Resets the device maps (if any) to None.
        """
        if self.hf_device_map is None:
            return
        else:
            self.remove_all_hooks()
            for name, component in self.components.items():
                if isinstance(component, torch.nn.Module):
                    component.to("cpu")
            self.hf_device_map = None

    @classmethod
    @validate_hf_hub_args
    def download(cls, pretrained_model_name, **kwargs) -> Union[str, os.PathLike]:
        r"""
        Download and cache a PyTorch diffusion pipeline from pretrained pipeline weights.

        Parameters:
            pretrained_model_name (`str` or `os.PathLike`, *optional*):
                A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                hosted on the Hub.
            custom_pipeline (`str`, *optional*):
                Can be either:

                    - A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained
                      pipeline hosted on the Hub. The repository must contain a file called `pipeline.py` that defines
                      the custom pipeline.

                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current `main` branch of GitHub.

                    - A path to a *directory* (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                For more information on how to load and create custom pipelines, take a look at [How to contribute a
                community pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline).

            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `False`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom pipelines and components defined on the Hub in their own files. This
                option should only be set to `True` for repositories you trust and in which you have read the code, as
                it will execute code present on the Hub on your local machine.

        Returns:
            `os.PathLike`:
                A path to the downloaded pipeline.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`.

        </Tip>

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        allow_patterns = None
        ignore_patterns = None

        model_info_call_error: Optional[Exception] = None
        if not local_files_only:
            try:
                info = model_info(pretrained_model_name, token=token, revision=revision)
            except (HTTPError, OfflineModeIsEnabled, requests.ConnectionError) as e:
                logger.warning(f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache.")
                local_files_only = True
                model_info_call_error = e  # save error to reraise it if model is not cached locally

        if not local_files_only:
            filenames = {sibling.rfilename for sibling in info.siblings}
            if variant is not None and _check_legacy_sharding_variant_format(filenames=filenames, variant=variant):
                warn_msg = (
                    f"Warning: The repository contains sharded checkpoints for variant '{variant}' maybe in a deprecated format. "
                    "Please check your files carefully:\n\n"
                    "- Correct format example: diffusion_pytorch_model.fp16-00003-of-00003.safetensors\n"
                    "- Deprecated format example: diffusion_pytorch_model-00001-of-00002.fp16.safetensors\n\n"
                    "If you find any files in the deprecated format:\n"
                    "1. Remove all existing checkpoint files for this variant.\n"
                    "2. Re-obtain the correct files by running `save_pretrained()`.\n\n"
                    "This will ensure you're using the most up-to-date and compatible checkpoint format."
                )
                logger.warning(warn_msg)

            model_filenames, variant_filenames = variant_compatible_siblings(filenames, variant=variant)

            config_file = hf_hub_download(
                pretrained_model_name,
                cls.config_name,
                cache_dir=cache_dir,
                revision=revision,
                proxies=proxies,
                force_download=force_download,
                token=token,
            )

            config_dict = cls._dict_from_json_file(config_file)
            ignore_filenames = config_dict.pop("_ignore_files", [])

            # remove ignored filenames
            model_filenames = set(model_filenames) - set(ignore_filenames)
            variant_filenames = set(variant_filenames) - set(ignore_filenames)

            if revision in DEPRECATED_REVISION_ARGS and version.parse(
                version.parse(__version__).base_version
            ) >= version.parse("0.22.0"):
                warn_deprecated_model_variant(pretrained_model_name, token, variant, revision, model_filenames)

            custom_components, folder_names = _get_custom_components_and_folders(
                pretrained_model_name, config_dict, filenames, variant_filenames, variant
            )
            model_folder_names = {os.path.split(f)[0] for f in model_filenames if os.path.split(f)[0] in folder_names}

            custom_class_name = None
            if custom_pipeline is None and isinstance(config_dict["_class_name"], (list, tuple)):
                custom_pipeline = config_dict["_class_name"][0]
                custom_class_name = config_dict["_class_name"][1]

            # all filenames compatible with variant will be added
            allow_patterns = list(model_filenames)

            # allow all patterns from non-model folders
            # this enables downloading schedulers, tokenizers, ...
            allow_patterns += [f"{k}/*" for k in folder_names if k not in model_folder_names]
            # add custom component files
            allow_patterns += [f"{k}/{f}.py" for k, f in custom_components.items()]
            # add custom pipeline file
            allow_patterns += [f"{custom_pipeline}.py"] if f"{custom_pipeline}.py" in filenames else []
            # also allow downloading config.json files with the model
            allow_patterns += [os.path.join(k, "config.json") for k in model_folder_names]

            allow_patterns += [
                SCHEDULER_CONFIG_NAME,
                CONFIG_NAME,
                cls.config_name,
                CUSTOM_PIPELINE_FILE_NAME,
            ]

            load_pipe_from_hub = custom_pipeline is not None and f"{custom_pipeline}.py" in filenames
            load_components_from_hub = len(custom_components) > 0

            if load_pipe_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {custom_pipeline}.py which must be executed to correctly "
                    f"load the model. You can inspect the repository content at https://hf.co/{pretrained_model_name}/blob/main/{custom_pipeline}.py.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            if load_components_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {'.py, '.join([os.path.join(k, v) for k,v in custom_components.items()])} which must be executed to correctly "
                    f"load the model. You can inspect the repository content at {', '.join([f'https://hf.co/{pretrained_model_name}/{k}/{v}.py' for k,v in custom_components.items()])}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            # retrieve passed components that should not be downloaded
            pipeline_class = _get_pipeline_class(
                cls,
                config_dict,
                load_connected_pipeline=load_connected_pipeline,
                custom_pipeline=custom_pipeline,
                repo_id=pretrained_model_name if load_pipe_from_hub else None,
                hub_revision=revision,
                class_name=custom_class_name,
                cache_dir=cache_dir,
                revision=custom_revision,
            )
            expected_components, _ = cls._get_signature_keys(pipeline_class)
            passed_components = [k for k in expected_components if k in kwargs]

            # retrieve all patterns that should not be downloaded and error out when needed
            ignore_patterns = _get_ignore_patterns(
                passed_components,
                model_folder_names,
                model_filenames,
                variant_filenames,
                use_safetensors,
                from_flax,
                allow_pickle,
                use_onnx,
                pipeline_class._is_onnx,
                variant,
            )

            # Don't download any objects that are passed
            allow_patterns = [
                p for p in allow_patterns if not (len(p.split("/")) == 2 and p.split("/")[0] in passed_components)
            ]

            if pipeline_class._load_connected_pipes:
                allow_patterns.append("README.md")

            # Don't download index files of forbidden patterns either
            ignore_patterns = ignore_patterns + [f"{i}.index.*json" for i in ignore_patterns]
            re_ignore_pattern = [re.compile(fnmatch.translate(p)) for p in ignore_patterns]
            re_allow_pattern = [re.compile(fnmatch.translate(p)) for p in allow_patterns]

            expected_files = [f for f in filenames if not any(p.match(f) for p in re_ignore_pattern)]
            expected_files = [f for f in expected_files if any(p.match(f) for p in re_allow_pattern)]

            snapshot_folder = Path(config_file).parent
            pipeline_is_cached = all((snapshot_folder / f).is_file() for f in expected_files)

            if pipeline_is_cached and not force_download:
                # if the pipeline is cached, we can directly return it
                # else call snapshot_download
                return snapshot_folder

        user_agent = {"pipeline_class": cls.__name__}
        if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
            user_agent["custom_pipeline"] = custom_pipeline

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = snapshot_download(
                pretrained_model_name,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )

            # retrieve pipeline class from local file
            cls_name = cls.load_config(os.path.join(cached_folder, "model_index.json")).get("_class_name", None)
            cls_name = cls_name[4:] if isinstance(cls_name, str) and cls_name.startswith("Flax") else cls_name

            diffusers_module = importlib.import_module(__name__.split(".")[0])
            pipeline_class = getattr(diffusers_module, cls_name, None) if isinstance(cls_name, str) else None

            if pipeline_class is not None and pipeline_class._load_connected_pipes:
                modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
                connected_pipes = sum([getattr(modelcard.data, k, []) for k in CONNECTED_PIPES_KEYS], [])
                for connected_pipe_repo_id in connected_pipes:
                    download_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "local_files_only": local_files_only,
                        "token": token,
                        "variant": variant,
                        "use_safetensors": use_safetensors,
                    }
                    DiffusionPipeline.download(connected_pipe_repo_id, **download_kwargs)

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise EnvironmentError(
                    f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        optional_names = list(optional_parameters)
        for name in optional_names:
            if name in cls._optional_components:
                expected_modules.add(name)
                optional_parameters.remove(name)

        return expected_modules, optional_parameters

    @classmethod
    def _get_signature_types(cls):
        signature_types = {}
        for k, v in inspect.signature(cls.__init__).parameters.items():
            if inspect.isclass(v.annotation):
                signature_types[k] = (v.annotation,)
            elif get_origin(v.annotation) == Union:
                signature_types[k] = get_args(v.annotation)
            else:
                logger.warning(f"cannot get type annotation for Parameter {k} of {cls}.")
        return signature_types

    @property
    def components(self) -> Dict[str, Any]:
        r"""
        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations without reallocating additional memory.

        Returns (`dict`):
            A dictionary containing all the modules needed to initialize the pipeline.

        Examples:

        ```py
        >>> from diffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        ```
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components.keys()} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        return numpy_to_pil(images)

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/). When this
        option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
        up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        >>> # Workaround for not accepting attention shape using VAE for Flash Attention
        >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            fn_recursive_set_mem_eff(module)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
        in slices to compute attention in several steps. For more than one attention head, the computation is performed
        sequentially over each head. This is useful to save some memory in exchange for a small speed decrease.

        <Tip warning={true}>

        ⚠️ Don't enable attention slicing if you're already using `scaled_dot_product_attention` (SDPA) from PyTorch
        2.0 or xFormers. These attention computations are already very memory efficient so you won't need to enable
        this function. If you enable attention slicing with SDPA or xFormers, it can lead to serious slow downs!

        </Tip>

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5",
        ...     torch_dtype=torch.float16,
        ...     use_safetensors=True,
        ... )

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> pipe.enable_attention_slicing()
        >>> image = pipe(prompt).images[0]
        ```
        """
        self.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously called, attention is
        computed in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def set_attention_slice(self, slice_size: Optional[int]):
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module) and hasattr(m, "set_attention_slice")]

        for module in modules:
            module.set_attention_slice(slice_size)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Create a new pipeline from a given pipeline. This method is useful to create a new pipeline from the existing
        pipeline components without reallocating additional memory.

        Arguments:
            pipeline (`DiffusionPipeline`):
                The pipeline from which to create a new pipeline.

        Returns:
            `DiffusionPipeline`:
                A new pipeline with the same weights and configurations as `pipeline`.

        Examples:

        ```py
        >>> from diffusers import StableDiffusionPipeline, StableDiffusionSAGPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> new_pipe = StableDiffusionSAGPipeline.from_pipe(pipe)
        ```
        """

        original_config = dict(pipeline.config)
        torch_dtype = kwargs.pop("torch_dtype", None)

        # derive the pipeline class to instantiate
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)

        if custom_pipeline is not None:
            pipeline_class = _get_custom_pipeline_class(custom_pipeline, revision=custom_revision)
        else:
            pipeline_class = cls

        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        # true_optional_modules are optional components with default value in signature so it is ok not to pass them to `__init__`
        # e.g. `image_encoder` for StableDiffusionPipeline
        parameters = inspect.signature(cls.__init__).parameters
        true_optional_modules = set(
            {k for k, v in parameters.items() if v.default != inspect._empty and k in expected_modules}
        )

        # get the class of each component based on its type hint
        # e.g. {"unet": UNet2DConditionModel, "text_encoder": CLIPTextMode}
        component_types = pipeline_class._get_signature_types()

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)
        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}

        original_class_obj = {}
        for name, component in pipeline.components.items():
            if name in expected_modules and name not in passed_class_obj:
                # for model components, we will not switch over if the class does not matches the type hint in the new pipeline's signature
                if (
                    not isinstance(component, ModelMixin)
                    or type(component) in component_types[name]
                    or (component is None and name in cls._optional_components)
                ):
                    original_class_obj[name] = component
                else:
                    logger.warning(
                        f"component {name} is not switched over to new pipeline because type does not match the expected."
                        f" {name} is {type(component)} while the new pipeline expect {component_types[name]}."
                        f" please pass the component of the correct type to the new pipeline. `from_pipe(..., {name}={name})`"
                    )

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k in original_config.keys()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config attribute that were not expected by pipeline is stored as its private attribute
        # (i.e. when the original pipeline was also instantiated with `from_pipe` from another pipeline that has this config)
        # in this case, we will pass them as optional arguments if they can be accepted by the new pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        pipeline_kwargs = {
            **passed_class_obj,
            **original_class_obj,
            **passed_pipe_kwargs,
            **original_pipe_kwargs,
            **kwargs,
        }

        # store unused config as private attribute in the new pipeline
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": v for k, v in original_config.items() if k not in pipeline_kwargs
        }

        missing_modules = (
            set(expected_modules)
            - set(pipeline._optional_components)
            - set(pipeline_kwargs.keys())
            - set(true_optional_modules)
        )

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        new_pipeline = pipeline_class(**pipeline_kwargs)
        if pretrained_model_name_or_path is not None:
            new_pipeline.register_to_config(_name_or_path=pretrained_model_name_or_path)
        new_pipeline.register_to_config(**unused_original_config)

        if torch_dtype is not None:
            new_pipeline.to(dtype=torch_dtype)

        return new_pipeline


class StableDiffusionMixin:
    r"""
    Helper for DiffusionPipeline with vae and unet.(mainly for LDM such as stable diffusion)
    """

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    def fuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_unet = False
        self.fusing_vae = False

        if unet:
            self.fusing_unet = True
            self.unet.fuse_qkv_projections()
            self.unet.set_attn_processor(FusedAttnProcessor2_0())

        if vae:
            if not isinstance(self.vae, AutoencoderKL):
                raise ValueError("`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.")

            self.fusing_vae = True
            self.vae.fuse_qkv_projections()
            self.vae.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if unet:
            if not self.fusing_unet:
                logger.warning("The UNet was not initially fused for QKV projections. Doing nothing.")
            else:
                self.unet.unfuse_qkv_projections()
                self.fusing_unet = False

        if vae:
            if not self.fusing_vae:
                logger.warning("The VAE was not initially fused for QKV projections. Doing nothing.")
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False
