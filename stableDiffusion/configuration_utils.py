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
"""ConfigMixin base class and utilities."""

import dataclasses
import functools
import importlib
import inspect
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
from huggingface_hub import create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from requests import HTTPError

from diffusers import __version__
from diffusers.utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    DummyObject,
    deprecate,
    extract_commit_hash,
    http_user_agent,
    logging,
)


logger = logging.get_logger(__name__)

_re_configuration_file = re.compile(r"config\.(.*)\.json")


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)


class ConfigMixin:
    r"""
    Base class for all configuration classes. All configuration parameters are stored under `self.config`. Also
    provides the [`~ConfigMixin.from_config`] and [`~ConfigMixin.save_config`] methods for loading, downloading, and
    saving classes that inherit from [`ConfigMixin`].

    Class attributes:
        - **config_name** (`str`) -- A filename under which the config should stored when calling
          [`~ConfigMixin.save_config`] (should be overridden by parent class).
        - **ignore_for_config** (`List[str]`) -- A list of attributes that should not be saved in the config (should be
          overridden by subclass).
        - **has_compatibles** (`bool`) -- Whether the class has compatible classes (should be overridden by subclass).
        - **_deprecated_kwargs** (`List[str]`) -- Keyword arguments that are deprecated. Note that the `init` function
          should only have a `kwargs` argument if at least one argument is deprecated (should be overridden by
          subclass).
    """

    config_name = None
    ignore_for_config = []
    has_compatibles = False

    _deprecated_kwargs = []

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def __getattr__(self, name: str) -> Any:
        """The only reason we overwrite `getattr` here is to gracefully deprecate accessing
        config attributes directly. See https://github.com/huggingface/diffusers/pull/3129

        This function is mostly copied from PyTorch's __getattr__ overwrite:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'scheduler.config.{name}'."
            deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)
            return self._internal_dict[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory specified in `save_directory` so that it can be reloaded using the
        [`~ConfigMixin.from_config`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file is saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    @classmethod
    def from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs):
        r"""
        Instantiate a Python class from a config dictionary.

        Parameters:
            config (`Dict[str, Any]`):
                A config dictionary from which the Python class is instantiated. Make sure to only load configuration
                files of compatible classes.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it is loaded) and initiate the Python class.
                `**kwargs` are passed directly to the underlying scheduler/model's `__init__` method and eventually
                overwrite the same named arguments in `config`.

        Returns:
            [`ModelMixin`] or [`SchedulerMixin`]:
                A model or scheduler object instantiated from a config dictionary.

        Examples:

        ```python
        >>> from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler

        >>> # Download scheduler from huggingface.co and cache.
        >>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

        >>> # Instantiate DDIM scheduler class with same config as DDPM
        >>> scheduler = DDIMScheduler.from_config(scheduler.config)

        >>> # Instantiate PNDM scheduler class with same config as DDPM
        >>> scheduler = PNDMScheduler.from_config(scheduler.config)
        ```
        """
        # <===== TO BE REMOVED WITH DEPRECATION
        # TODO(Patrick) - make sure to remove the following lines when config=="model_path" is deprecated
        if "pretrained_model_name_or_path" in kwargs:
            config = kwargs.pop("pretrained_model_name_or_path")

        if config is None:
            raise ValueError("Please make sure to provide a config as the first positional argument.")
        # ======>

        if not isinstance(config, dict):
            deprecation_message = "It is deprecated to pass a pretrained model name or path to `from_config`."
            if "Scheduler" in cls.__name__:
                deprecation_message += (
                    f"If you were trying to load a scheduler, please use {cls}.from_pretrained(...) instead."
                    " Otherwise, please make sure to pass a configuration dictionary instead. This functionality will"
                    " be removed in v1.0.0."
                )
            elif "Model" in cls.__name__:
                deprecation_message += (
                    f"If you were trying to load a model, please use {cls}.load_config(...) followed by"
                    f" {cls}.from_config(...) instead. Otherwise, please make sure to pass a configuration dictionary"
                    " instead. This functionality will be removed in v1.0.0."
                )
            deprecate("config-passed-as-path", "1.0.0", deprecation_message, standard_warn=False)
            config, kwargs = cls.load_config(pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)

        # Allow dtype to be specified on initialization
        if "dtype" in unused_kwargs:
            init_dict["dtype"] = unused_kwargs.pop("dtype")

        # add possible deprecated kwargs
        for deprecated_kwarg in cls._deprecated_kwargs:
            if deprecated_kwarg in unused_kwargs:
                init_dict[deprecated_kwarg] = unused_kwargs.pop(deprecated_kwarg)

        # Return model and optionally state and/or unused_kwargs
        model = cls(**init_dict)

        # make sure to also save config parameters that might be used for compatible classes
        # update _class_name
        if "_class_name" in hidden_dict:
            hidden_dict["_class_name"] = cls.__name__

        model.register_to_config(**hidden_dict)

        # add hidden kwargs of compatible classes to unused_kwargs
        unused_kwargs = {**unused_kwargs, **hidden_dict}

        if return_unused_kwargs:
            return (model, unused_kwargs)
        else:
            return model

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        deprecation_message = (
            f" The function get_config_dict is deprecated. Please use {cls}.load_config instead. This function will be"
            " removed in version v1.0.0"
        )
        deprecate("get_config_dict", "1.0.0", deprecation_message, standard_warn=False)
        return cls.load_config(*args, **kwargs)

    @classmethod
    @validate_hf_hub_args
    def load_config(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        return_unused_kwargs=False,
        return_commit_hash=False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""
        Load a model or scheduler configuration.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing model weights saved with
                      [`~ConfigMixin.save_config`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
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
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False):
                Whether unused keyword arguments of the config are returned.
            return_commit_hash (`bool`, *optional*, defaults to `False):
                Whether the `commit_hash` of the loaded configuration are returned.

        Returns:
            `dict`:
                A dictionary of all the parameters stored in a JSON configuration file.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", None)
        user_agent = kwargs.pop("user_agent", {})
    # def ok23234():
        print("\n[Config Preparation] 开始配置准备流程")
        
        # 参数提取阶段
        print("\n[阶段1] 提取配置参数")
        param_table = [
            ("cache_dir", "缓存目录", None),
            ("local_dir", "本地目录", None),
            ("local_dir_use_symlinks", "符号链接策略", "auto"),
            ("force_download", "强制下载", False),
            ("proxies", "代理设置", None),
            ("token", "访问令牌", None),
            ("local_files_only", "仅本地文件", False),
            ("revision", "版本修订", None),
            ("subfolder", "子目录路径", None),
        ]
        
        extracted_params = {}
        for param, desc, default in param_table:
            value = kwargs.pop(param, default)
            status = "默认值" if value == default else "自定义值"
            extracted_params[param] = value
            print(f"   → {param.ljust(20)}: {str(value).ljust(15)} | {status} ({desc})")
        
        # 处理特殊参数
        print("\n[阶段2] 处理元数据参数")
        print("🔄 合并用户代理信息:")
        print(f"   原始 user_agent: {kwargs.get('user_agent', {})}")
        user_agent = kwargs.pop("user_agent", {})
        user_agent.update({"file_type": "config"})
        print(f"   合并后 user_agent: {user_agent}")
        
        final_user_agent = http_user_agent(user_agent)
        print(f"✅ 最终 User-Agent 字符串: {final_user_agent}")
        
        # 镜像参数处理
        mirror = kwargs.pop("mirror", None)
        if mirror:
            print(f"⚠️ 检测到镜像参数已忽略: mirror={mirror}")

        # 路径处理
        print("\n[阶段3] 处理模型路径")
        original_path = pretrained_model_name_or_path
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        print(f"   → 原始路径类型: {type(original_path).__name__}")
        print(f"   → 转换后路径: {pretrained_model_name_or_path}")

        # 配置名称检查
        print("\n[阶段4] 验证配置完整性")
        if cls.config_name is None:
            error_msg = (
                "关键配置缺失: ConfigMixin 未定义 config_name\n"
                "问题诊断:\n"
                "1. 请确保在继承 ConfigMixin 的子类中定义 config_name\n"
                "2. 检查类定义是否包含: config_name = 'your_config_name'"
            )
            print("❌ " + "\n".join(error_msg.split("\n")[:2]))
            raise ValueError(error_msg)
        else:
            print(f"✅ 配置标识验证通过 | config_name='{cls.config_name}'")

        # print("\n[Config Preparation] 配置准备完成 ✅\n")

    # def ok234252():
        print("\n[Config Detection] 开始配置文件路径检测")
        
        print(f"🔍 输入路径分析: {pretrained_model_name_or_path}")

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            ):
                config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.config_name)):
                # Load from a PyTorch checkpoint
                config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `token` or log in with `huggingface-cli login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
                )
            except HTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(config_file)

            commit_hash = extract_commit_hash(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")

        if not (return_unused_kwargs or return_commit_hash):
            return config_dict

        outputs = (config_dict,)

        if return_unused_kwargs:
            outputs += (kwargs,)

        if return_commit_hash:
            outputs += (commit_hash,)

        return outputs

    # @classmethod
    # @validate_hf_hub_args
    def load_config1(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        return_unused_kwargs=False,
        return_commit_hash=False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""
        Load a model or scheduler configuration.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing model weights saved with
                      [`~ConfigMixin.save_config`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
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
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False):
                Whether unused keyword arguments of the config are returned.
            return_commit_hash (`bool`, *optional*, defaults to `False):
                Whether the `commit_hash` of the loaded configuration are returned.

        Returns:
            `dict`:
                A dictionary of all the parameters stored in a JSON configuration file.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", None)
        user_agent = kwargs.pop("user_agent", {})
    # def ok23234():
        print("\n[Config Preparation] 开始配置准备流程")
        
        # 参数提取阶段
        print("\n[阶段1] 提取配置参数")
        param_table = [
            ("cache_dir", "缓存目录", None),
            ("local_dir", "本地目录", None),
            ("local_dir_use_symlinks", "符号链接策略", "auto"),
            ("force_download", "强制下载", False),
            ("proxies", "代理设置", None),
            ("token", "访问令牌", None),
            ("local_files_only", "仅本地文件", False),
            ("revision", "版本修订", None),
            ("subfolder", "子目录路径", None),
        ]
        
        extracted_params = {}
        for param, desc, default in param_table:
            value = kwargs.pop(param, default)
            status = "默认值" if value == default else "自定义值"
            extracted_params[param] = value
            print(f"   → {param.ljust(20)}: {str(value).ljust(15)} | {status} ({desc})")
        
        # 处理特殊参数
        print("\n[阶段2] 处理元数据参数")
        print("🔄 合并用户代理信息:")
        print(f"   原始 user_agent: {kwargs.get('user_agent', {})}")
        user_agent = kwargs.pop("user_agent", {})
        user_agent.update({"file_type": "config"})
        print(f"   合并后 user_agent: {user_agent}")
        
        final_user_agent = http_user_agent(user_agent)
        print(f"✅ 最终 User-Agent 字符串: {final_user_agent}")
        
        # 镜像参数处理
        mirror = kwargs.pop("mirror", None)
        if mirror:
            print(f"⚠️ 检测到镜像参数已忽略: mirror={mirror}")

        # 路径处理
        print("\n[阶段3] 处理模型路径")
        original_path = pretrained_model_name_or_path
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        print(f"   → 原始路径类型: {type(original_path).__name__}")
        print(f"   → 转换后路径: {pretrained_model_name_or_path}")

        # 配置名称检查
        print("\n[阶段4] 验证配置完整性")
        if cls.config_name is None:
            error_msg = (
                "关键配置缺失: ConfigMixin 未定义 config_name\n"
                "问题诊断:\n"
                "1. 请确保在继承 ConfigMixin 的子类中定义 config_name\n"
                "2. 检查类定义是否包含: config_name = 'your_config_name'"
            )
            print("❌ " + "\n".join(error_msg.split("\n")[:2]))
            raise ValueError(error_msg)
        else:
            print(f"✅ 配置标识验证通过 | config_name='{cls.config_name}'")

        # print("\n[Config Preparation] 配置准备完成 ✅\n")

    # def ok234252():
        print("\n[Config Detection] 开始配置文件路径检测")
        
        print(f"🔍 输入路径分析: {pretrained_model_name_or_path}")
        
        # 路径类型检查
        if os.path.isfile(pretrained_model_name_or_path):
            print("✅ 检测到直接配置文件路径")
            print(f"   → 文件类型: {os.path.splitext(pretrained_model_name_or_path)[1]}")
            config_file = pretrained_model_name_or_path
            print(f"📄 使用配置文件: {config_file}")
            return config_file

        elif os.path.isdir(pretrained_model_name_or_path):
            print("📂 输入路径为目录，开始搜索配置文件")
            # print(f"   → 指定子目录参数: {subfolder or '未指定'}")
            
            # 子目录检查
            subfolder_path = os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else None
            if subfolder_path and os.path.exists(subfolder_path):
                print(f"🔎 检查子目录: {subfolder}")
                target_path = os.path.join(subfolder_path, cls.config_name)
                if os.path.isfile(target_path):
                    print(f"✅ 在子目录找到配置文件")
                    config_file = target_path
                    print(f"📄 配置文件路径: {config_file}")
                    return config_file

            # 主目录检查
            print("🔎 检查主目录配置文件")
            main_config_path = os.path.join(pretrained_model_name_or_path, cls.config_name)
            if os.path.isfile(main_config_path):
                print(f"✅ 在主目录找到配置文件")
                config_file = main_config_path
                print(f"📄 配置文件路径: {config_file}")
                return config_file

            # 路径诊断信息
            checked_paths = []
            if subfolder:
                checked_paths.append(f"子目录: {os.path.join(pretrained_model_name_or_path, subfolder)}")
            checked_paths.append(f"主目录: {pretrained_model_name_or_path}")
            
            error_msg = (
                f"\n❌ 配置文件 {cls.config_name} 未找到\n"
                f"已检查路径:\n" + 
                "\n".join([f"   → {path}" for path in checked_paths]) +
                f"\n可能原因:\n"
                f"1. 模型目录结构不正确\n"
                f"2. 配置文件名称不匹配 (当前配置名: {cls.config_name})\n"
                f"3. 未正确指定子目录参数 (当前subfolder: {subfolder})"
            )
            print(error_msg)
            raise EnvironmentError(error_msg)

        # else:
        #     error_msg = (
        #         f"\n❌ 无效路径类型\n"
        #         f"输入路径: {pretrained_model_name_or_path}\n"
        #         f"路径类型: {'文件' if os.path.isfile(pretrained_model_name_or_path) else '目录' if os.path.isdir(pretrained_model_name_or_path) else '不存在'}\n"
        #         f"建议操作:\n"
        #         f"1. 检查路径拼写是否正确\n"
        #         f"2. 确认模型文件已下载\n"
        #         f"3. 验证文件系统权限"
        #     )
        #     print(error_msg)
        #     raise EnvironmentError(f"Invalid path: {pretrained_model_name_or_path}")

        # print("\n[Config Detection] 路径检测完成\n")

        else:
            try:
                # Load from URL or cache if already cached
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `token` or log in with `huggingface-cli login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
                )
            except HTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )
    # def ok243242():
        print("\n[Config Loading] 开始配置文件加载流程")
        
        try:
            print(f"📄 正在解析配置文件: {config_file}")
            print("🔍 尝试加载JSON配置...")
            config_dict = cls._dict_from_json_file(config_file)
            print(f"✅ JSON解析成功 | 包含 {len(config_dict)} 个配置项")
            
            print("\n🔄 提取Git提交哈希")
            commit_hash = extract_commit_hash(config_file)
            if commit_hash:
                print(f"   → 提交哈希: {commit_hash[:7]}...")
            else:
                print("⚠️ 未找到版本提交信息")
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"\n❌ 配置文件解析失败: {str(e)}")
            print("▌" + " 错误诊断 ".center(50, '─'))
            print("│ 可能原因:")
            print("│ 1. 文件内容不是合法JSON格式")
            print("│ 2. 文件编码非UTF-8")
            print(f"│ 3. 文件可能已损坏 (路径: {config_file})")
            print("└" + "─"*50)
            raise EnvironmentError(f"Invalid config file at '{config_file}'")

        # 构建返回结果
        return_params = []
        print("\n[阶段] 构建返回参数")
        print(f"   → return_unused_kwargs: {return_unused_kwargs}")
        print(f"   → return_commit_hash: {return_commit_hash}")
        
        outputs = (config_dict,)
        return_params.append(f"配置字典({len(config_dict)}项)")

        if return_unused_kwargs:
            print(f"➕ 添加未使用参数 | 数量: {len(kwargs)}")
            outputs += (kwargs,)
            return_params.append(f"未使用参数({len(kwargs)}个)")
            
        if return_commit_hash:
            print(f"➕ 添加版本哈希: {commit_hash[:7] if commit_hash else 'None'}")
            outputs += (commit_hash,)
            return_params.append("提交哈希")

        print(f"\n📤 返回参数组合: {', '.join(return_params)}")
        print("[Config Loading] 配置加载完成 ✅\n")
        
        # return outputs if len(outputs) > 1 else outputs[0]


        return outputs

    @staticmethod
    def _get_init_keys(input_class):
        return set(dict(inspect.signature(input_class.__init__).parameters).keys())

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        # Skip keys that were not present in the original config, so default __init__ values were used
        used_defaults = config_dict.get("_use_default_values", [])
        config_dict = {k: v for k, v in config_dict.items() if k not in used_defaults and k != "_use_default_values"}

        # 0. Copy origin config dict
        original_dict = dict(config_dict.items())

        # 1. Retrieve expected config attributes from __init__ signature
        expected_keys = cls._get_init_keys(cls)
        expected_keys.remove("self")
        # remove general kwargs if present in dict
        if "kwargs" in expected_keys:
            expected_keys.remove("kwargs")
        # remove flax internal keys
        if hasattr(cls, "_flax_internal_args"):
            for arg in cls._flax_internal_args:
                expected_keys.remove(arg)

        # 2. Remove attributes that cannot be expected from expected config attributes
        # remove keys to be ignored
        if len(cls.ignore_for_config) > 0:
            expected_keys = expected_keys - set(cls.ignore_for_config)

        # load diffusers library to import compatible and original scheduler
        diffusers_library = importlib.import_module(__name__.split(".")[0])

        if cls.has_compatibles:
            compatible_classes = [c for c in cls._get_compatibles() if not isinstance(c, DummyObject)]
        else:
            compatible_classes = []

        expected_keys_comp_cls = set()
        for c in compatible_classes:
            expected_keys_c = cls._get_init_keys(c)
            expected_keys_comp_cls = expected_keys_comp_cls.union(expected_keys_c)
        expected_keys_comp_cls = expected_keys_comp_cls - cls._get_init_keys(cls)
        config_dict = {k: v for k, v in config_dict.items() if k not in expected_keys_comp_cls}

        # remove attributes from orig class that cannot be expected
        orig_cls_name = config_dict.pop("_class_name", cls.__name__)
        if (
            isinstance(orig_cls_name, str)
            and orig_cls_name != cls.__name__
            and hasattr(diffusers_library, orig_cls_name)
        ):
            orig_cls = getattr(diffusers_library, orig_cls_name)
            unexpected_keys_from_orig = cls._get_init_keys(orig_cls) - expected_keys
            config_dict = {k: v for k, v in config_dict.items() if k not in unexpected_keys_from_orig}
        elif not isinstance(orig_cls_name, str) and not isinstance(orig_cls_name, (list, tuple)):
            raise ValueError(
                "Make sure that the `_class_name` is of type string or list of string (for custom pipelines)."
            )

        # remove private attributes
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

        # remove quantization_config
        config_dict = {k: v for k, v in config_dict.items() if k != "quantization_config"}

        # 3. Create keyword arguments that will be passed to __init__ from expected keyword arguments
        init_dict = {}
        for key in expected_keys:
            # if config param is passed to kwarg and is present in config dict
            # it should overwrite existing config dict key
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)

            if key in kwargs:
                # overwrite key
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                # use value from config dict
                init_dict[key] = config_dict.pop(key)

        # 4. Give nice warning if unexpected values have been passed
        if len(config_dict) > 0:
            logger.warning(
                f"The config attributes {config_dict} were passed to {cls.__name__}, "
                "but are not expected and will be ignored. Please verify your "
                f"{cls.config_name} configuration file."
            )

        # 5. Give nice info if config attributes are initialized to default because they have not been passed
        passed_keys = set(init_dict.keys())
        if len(expected_keys - passed_keys) > 0:
            logger.info(
                f"{expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
            )

        # 6. Define unused keyword arguments
        unused_kwargs = {**config_dict, **kwargs}

        # 7. Define "hidden" config parameters that were saved for compatible classes
        hidden_config_dict = {k: v for k, v in original_dict.items() if k not in init_dict}

        return init_dict, unused_kwargs, hidden_config_dict

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the config of the class as a frozen dictionary

        Returns:
            `Dict[str, Any]`: Config of the class.
        """
        return self._internal_dict

    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = self.__class__.__name__
        config_dict["_diffusers_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, Path):
                value = value.as_posix()
            return value

        if "quantization_config" in config_dict:
            config_dict["quantization_config"] = (
                config_dict.quantization_config.to_dict()
                if not isinstance(config_dict.quantization_config, dict)
                else config_dict.quantization_config
            )

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        # Don't save "_ignore_files" or "_use_default_values"
        config_dict.pop("_ignore_files", None)
        config_dict.pop("_use_default_values", None)
        # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
        _ = config_dict.pop("_pre_quantization_dtype", None)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save the configuration instance's parameters to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file to save a configuration instance's parameters.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


def register_to_config(init):
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )

        # Take note of the parameters that were not present in the loaded config
        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))

        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init


def flax_register_to_config(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        # Ignore private kwargs in the init. Retrieve all passed attributes
        init_kwargs = dict(kwargs.items())

        # Retrieve default values
        fields = dataclasses.fields(self)
        default_kwargs = {}
        for field in fields:
            # ignore flax specific attributes
            if field.name in self._flax_internal_args:
                continue
            if type(field.default) == dataclasses._MISSING_TYPE:
                default_kwargs[field.name] = None
            else:
                default_kwargs[field.name] = getattr(self, field.name)

        # Make sure init_kwargs override default kwargs
        new_kwargs = {**default_kwargs, **init_kwargs}
        # dtype should be part of `init_kwargs`, but not `new_kwargs`
        if "dtype" in new_kwargs:
            new_kwargs.pop("dtype")

        # Get positional arguments aligned with kwargs
        for i, arg in enumerate(args):
            name = fields[i].name
            new_kwargs[name] = arg

        # Take note of the parameters that were not present in the loaded config
        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))

        getattr(self, "register_to_config")(**new_kwargs)
        original_init(self, *args, **kwargs)

    cls.__init__ = init
    return cls


class LegacyConfigMixin(ConfigMixin):
    r"""
    A subclass of `ConfigMixin` to resolve class mapping from legacy classes (like `Transformer2DModel`) to more
    pipeline-specific classes (like `DiTTransformer2DModel`).
    """

    @classmethod
    def from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs):
        # To prevent dependency import problem.
        from diffusers.models.model_loading_utils import _fetch_remapped_cls_from_config

        # resolve remapping
        remapped_class = _fetch_remapped_cls_from_config(config, cls)

        return remapped_class.from_config(config, return_unused_kwargs, **kwargs)
