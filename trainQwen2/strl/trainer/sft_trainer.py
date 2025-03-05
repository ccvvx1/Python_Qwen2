# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import dataclasses
import os
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Type, Union

import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorForLanguageModeling,
    FeatureExtractionMixin,
    PreTrainedModel,
    # PreTrainedTokenizerBase,
    ProcessorMixin,
    # Trainer,
    TrainingArguments,
    is_wandb_available,
)
from llama.tokenization_llama_fast import LlamaTokenizerFast
from tokenization_utils_base import PreTrainedTokenizerBase
from trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from ..data_utils import is_conversational, maybe_apply_chat_template, maybe_convert_to_chatml, pack_examples
from .sft_config import SFTConfig
from .utils import ConstantLengthDataset, generate_model_card, get_comet_experiment_url, peft_module_casting_to_bf16


if is_peft_available():
    import speft
    from speft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

if is_wandb_available():
    import wandb

bPrintMoreKv = True
class SFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from datasets import load_dataset
    from trl import SFTTrainer

    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`SFTConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the prcessed `train_dataset` or `eval_dataset`.
            Will default to [`~transformers.default_data_collator`] if no `processing_class` is provided, an instance
            of [`~transformers.DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or
            tokenizer.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. SFT supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain an `input_ids` field.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            A tuple containing the optimizer class and keyword arguments to use.
            Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*, defaults to `None`):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Optional[Callable]`):
            Formatting function applied to the dataset before tokenization.
    """

    _tag_names = ["trl", "sft"]

    @deprecate_kwarg(
        "tokenizer", "0.16.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[Type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[dict], str], Callable[[dict], list[str]]]] = None,
    ):
        print("\n" + "="*40)
        print("Initializing SFT Trainer".center(40))
        print("="*40)
        
        # Args initialization
        print("\n[Phase 1] 参数初始化".ljust(40, '-'))
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            short_name = model_name.split("/")[-1]
            args = SFTConfig(f"{short_name}-SFT")
            print(f"自动生成SFT配置 | 模型简称: {short_name} | 配置名称: {args.output_dir}")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)
            print(f"转换TrainingArguments到SFTConfig | 原始参数数量: {len(dict_args)} | 转换后参数示例: {list(dict_args.keys())[:3]}...")
        
        # Model initialization
        print("\n[Phase 2] 模型初始化".ljust(40, '-'))
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn("model_init_kwargs将被忽略（模型已实例化）")
            print(f"⚠️ 警告 | 忽略model_init_kwargs参数: {list(args.model_init_kwargs.keys())}")
            
        if isinstance(model, str):
            print(f"从路径加载模型 | 模型标识: {model}")
            model = self._create_model_from_path(model, args)
            print(f"✅ 模型加载完成 | 类型: {type(model).__name__} | 参数量: {sum(p.numel() for p in model.parameters()):,}")
        else:
            print(f"使用已有模型实例 | 类型: {type(model).__name__} | 是否冻结参数: {all(not p.requires_grad for p in model.parameters())}")
        
        # PEFT configuration
        if peft_config is not None:
            print("\n[Phase 3] PEFT配置".ljust(40, '-'))
            print(f"应用PEFT配置前模型结构: {model}")
            print(f"PEFT配置详情: {peft_config.to_dict()}")
            model = self._prepare_peft_model(model, peft_config, args)
            # print(f"✅ PEFT应用完成 | 新模型结构: {type(model).__name__} | 可训练参数占比: {self._get_trainable_parameter_ratio(model):.1%}")
        
        # Tokenizer handling
        print("\n[Phase 4] 分词器处理".ljust(40, '-'))
        if processing_class is None:
            model_path = model.config._name_or_path if hasattr(model, 'config') else 'Unknown'
            print(f"自动加载分词器 | 模型路径: {model_path}")
            processing_class = LlamaTokenizerFast.from_pretrained(model_path)
            if processing_class.pad_token is None:
                processing_class.pad_token = processing_class.eos_token
                print(f"⚠️ 设置pad_token为eos_token | pad_token_id: {processing_class.pad_token_id}")
            print(f"分词器类型: {type(processing_class).__name__} | 词汇量: {processing_class.vocab_size:,}")
        else:
            print(f"使用自定义处理器 | 类型: {type(processing_class).__name__}")
        
        # Dataset preparation
        print("\n[Phase 5] 数据集准备".ljust(40, '-'))
        preprocess_dataset = args.dataset_kwargs is None or not args.dataset_kwargs.get("skip_prepare_dataset", False)

        # 数据集预处理逻辑
        print("\n[Phase 6] 数据集预处理".ljust(40, '-'))
        if preprocess_dataset:
            print(f"开始训练集预处理 | 原始数据集类型: {type(train_dataset).__name__}")
            original_train_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 'Unknown'
            
            # 进行数据预处理
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            
            new_train_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 'Unknown'
            print(f"✅ 训练集预处理完成 | 处理前样本数: {original_train_size} -> 处理后: {new_train_size}")
            
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                print(f"\n评估集打包策略: {'启用' if packing else '禁用'}")
                
                if isinstance(eval_dataset, dict):
                    print(f"处理字典格式评估集 | 包含{len(eval_dataset)}个子集: {list(eval_dataset.keys())}")
                    for key in eval_dataset:
                        original_eval_size = len(eval_dataset[key]) if hasattr(eval_dataset[key], '__len__') else 'Unknown'
                        eval_dataset[key] = self._prepare_dataset(
                            eval_dataset[key], processing_class, args, packing, formatting_func, key
                        )
                        new_eval_size = len(eval_dataset[key]) if hasattr(eval_dataset[key], '__len__') else 'Unknown'
                        print(f"  {key}子集处理完成 | {original_eval_size} -> {new_eval_size}")
                else:
                    original_eval_size = len(eval_dataset) if hasattr(eval_dataset, '__len__') else 'Unknown'
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )
                    new_eval_size = len(eval_dataset) if hasattr(eval_dataset, '__len__') else 'Unknown'
                    print(f"评估集处理完成 | {original_eval_size} -> {new_eval_size}")
        else:
            print("⚠️ 跳过数据集预处理步骤")

        # 数据整理器初始化
        print("\n[Phase 7] 数据整理器配置".ljust(40, '-'))
        if data_collator is None:
            print("自动创建默认数据整理器")
            print(f"分词器类型: {type(processing_class).__name__} | MLM: {'否'}")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=processing_class, 
                mlm=False
            )
            print(f"✅ 数据整理器创建完成 | 类型: {type(data_collator).__name__}")
        else:
            print(f"使用自定义数据整理器 | 类型: {type(data_collator).__name__}")

        self._metrics = defaultdict(list)

        # 父类初始化兼容性处理
        print("\n[Phase 8] 父类初始化准备".ljust(40, '-'))
        super_init_kwargs = {}
        transformers_version = version.parse(transformers.__version__)
        required_version = version.parse("4.47.0.dev0")
        
        print(f"检测Transformers版本: {transformers_version}")
        if transformers_version >= required_version:
            print(f"版本兼容 (≥{required_version}) | 注入优化器配置")
            super_init_kwargs["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs
            if optimizer_cls_and_kwargs:
                opt_cls, opt_kwargs = optimizer_cls_and_kwargs
                print(f"自定义优化器配置: {opt_cls.__name__} | 参数示例: {list(opt_kwargs.keys())[:2]}")
        else:
            if optimizer_cls_and_kwargs is not None:
                warnings.warn("旧版本Transformers不支持optimizer_cls_and_kwargs")
                print(f"⚠️ 版本过低 ({transformers_version} < {required_version}) | 已忽略优化器配置")

        # 调用父类初始化
        print("\n[Phase 9] 执行父类初始化".ljust(40, '-'))
        print("传递关键参数列表:")
        print(f"  - 模型类型: {type(model).__name__}")
        print(f"  - 训练集样本数: {len(train_dataset) if train_dataset else 'None'}")
        print(f"  - 回调函数数量: {len(callbacks) if callbacks else 0}")
        
        print("调用super().__init__...")
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **super_init_kwargs,
        )
        print("✅ 父类初始化完成")

        # 模型标签处理
        print("\n[Phase 10] 模型元数据配置".ljust(40, '-'))
        print(f"当前模型标签: {getattr(self, '_tag_names', '未定义')}")
        if hasattr(self.model, "add_model_tags"):
            print("检测到模型标签接口，添加训练方法标签")
            # self.model.add_model_tag("sft")
            # self.model.add_model_tag("transformers_trainer")
            self.model.add_model_tags(self._tag_names)
            print(f"更新后标签: {self.model.__class__.__name__} 的标签列表")
        else:
            print("⚠️ 模型不支持标签添加功能")


    def _create_model_from_path(self, model_path: str, args: SFTConfig) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Create model
        if args.use_liger:
            if not is_liger_kernel_available():
                raise ImportError("Please install Liger-kernel for use_liger=True")
            model = AutoLigerKernelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def _prepare_peft_model(self, model: PreTrainedModel, peft_config: Any, args: SFTConfig) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        if not is_peft_available():
            raise ImportError("To use PeftModel, you need to install the `peft` library.")

        if not isinstance(peft_config, PeftConfig):
            print(" ====期望的类型==== ", PeftConfig)
            raise ValueError(
                f"Expected PeftConfig object but got {type(peft_config)}. If you want to use the PeftModel, you need "
                "to pass a PeftConfig object to the SFTTrainer."
            )

        if isinstance(model, PeftModel):
            return model

        # Handle quantized models (QLoRA)
        # 是否进行低精度训练
        is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        is_sharded_qlora = False
        if getattr(model, "is_loaded_in_4bit", False):
            # Check if model is sharded (FSDP/DS-Zero3)
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                    break

        # Prepare model for kbit training if needed
        if is_qlora and not is_sharded_qlora:
            print("进行低精度训练")
            model = self._prepare_model_for_kbit_training(model, args)
            # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
            args = dataclasses.replace(args, gradient_checkpointing=False)
        elif args.gradient_checkpointing:
            print("进行高精度训练")
            model = self._enable_gradient_checkpointing(model, args)

        # Create PEFT model
        if (
            version.parse(speft.__version__) >= version.parse("0.12")  # autocast_adapter_dtype introduced in 0.12
            and getattr(model, "is_loaded_in_4bit", False)
            and is_sharded_qlora
        ):
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
            print("生成低精度微调模型")
        else:
            model = get_peft_model(model, peft_config)
            print("生成高精度微调模型")
            print(f"新的模型架构：{model}")

        # Handle bf16 casting for 4-bit models
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
            peft_module_casting_to_bf16(model)

        return model

    def _prepare_model_for_kbit_training(self, model: PreTrainedModel, args: SFTConfig) -> PreTrainedModel:
        """Prepares a quantized model for kbit training."""
        prepare_model_kwargs = {
            "use_gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_kwargs": args.gradient_checkpointing_kwargs or {},
        }

        return prepare_model_for_kbit_training(model, **prepare_model_kwargs)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: SFTConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
    # def ok123():
        print("🔧 开始配置梯度检查点与梯度需求设置")
        
        # 获取梯度检查点参数配置
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        print(f"📦 梯度检查点参数: {gradient_checkpointing_kwargs} (默认使用空字典)")

        # 确定检查点模式
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs 
            or gradient_checkpointing_kwargs["use_reentrant"]
        )
        print(f"🔍 检查use_reentrant标志: {'未指定' if 'use_reentrant' not in gradient_checkpointing_kwargs else '显式设置'}")
        print(f"🔄 使用{'可重入(reentrant)' if use_reentrant else '不可重入(non-reentrant)'}检查点模式")

        if use_reentrant:
            print("\n⚠️ 检测到需要启用输入梯度保留(reentrant模式)")
            
            if hasattr(model, "enable_input_require_grads"):
                print("✅ 检测到模型内置enable_input_require_grads方法")
                print("⚡ 正在启用自动输入梯度需求...")
                model.enable_input_require_grads()
                print("🟢 自动梯度需求配置完成")
            else:
                print("⛔ 模型未实现enable_input_require_grads方法，启用备用方案")
                
                def make_inputs_require_grad(module, input, output):
                    print(f"🎯 手动设置梯度需求 - 模块: {module._get_name()}")
                    output.requires_grad_(True)
                    
                print("🪝 注册输入嵌入层前向钩子...")
                embed_layer = model.get_input_embeddings()
                print(f"🔗 目标嵌入层: {embed_layer.__class__.__name__}[in={embed_layer.num_embeddings}, dim={embed_layer.embedding_dim}]")
                hook_handle = embed_layer.register_forward_hook(make_inputs_require_grad)
                print(f"📌 钩子注册成功 (句柄ID: {id(hook_handle)})")

        print("\n🔚 完成梯度检查点配置")
        print("="*50)
        return model



    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        print(f"\n🔍 开始处理数据集: {dataset_name} (类型: {type(dataset).__name__})")
        
        # 1. 处理 ConstantLengthDataset 类型
        if isinstance(dataset, ConstantLengthDataset):
            print("🔄 检测到 ConstantLengthDataset，跳过预处理步骤")
            return dataset

        # 2. 检查是否已预处理
        first_sample = next(iter(dataset))
        column_names = list(first_sample.keys())
        is_processed = "input_ids" in column_names
        print(f"📊 数据集列名: {column_names}")
        print(f"🔄 预处理状态: {'已处理' if is_processed else '未处理'}")

        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc
            print(f"⚙️ 设置多进程数: {args.dataset_num_proc}")

        with PartialState().local_main_process_first():
            # 3. 应用格式化函数
            if formatting_func is not None:
                if is_processed:
                    warnings.warn("忽略已处理数据集的格式化函数", UserWarning)
                    print("⚠️ 警告：检测到已处理数据集，跳过 formatting_func")
                else:
                    print(f"🛠️ 正在应用格式化函数: {formatting_func.__name__}")
                    batched = isinstance(formatting_func(first_sample), list)
                    print(f"📦 批处理模式: {'开启' if batched else '关闭'}")

                    def _func(example):
                        result = {"text": formatting_func(example)}
                        print(f"📝 格式化样例 -> 输入keys: {example.keys()} | 输出keys: {result.keys()}")
                        return result

                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"格式化 {dataset_name}"
                    dataset = dataset.map(_func, batched=batched, **map_kwargs)
                    print(f"✅ 格式化完成，新列: {dataset.column_names}")

            # 4. 合并 prompt/completion 字段
            if "prompt" in column_names and "completion" in column_names:
                print("🔀 检测到 prompt-completion 结构，开始合并...")
                key = "messages" if is_conversational(first_sample) else "text"
                print(f"🗝️ 合并后的字段名: {key}")

                def concat_prompt_completion(example):
                    merged = {key: example["prompt"] + example["completion"]}
                    print(f"✂️ 合并样例 -> 长度: {len(merged[key])} 字符")
                    return merged

                dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])
                print(f"✅ 合并完成，剩余列: {dataset.column_names}")

        global bPrintMoreKv  # 用于控制详细调试输出的全局变量
        
        # --- 转换到ChatML格式 ---
        print(f"\n=== 阶段1：开始转换数据集到ChatML格式 ===")
        if "conversations" in dataset.column_names:
            print(f"检测到原始对话列[conversations]，将执行格式转换")
        else:
            print(f"未检测到原始对话列，跳过转换")

        if isinstance(dataset, Dataset):  # 标准数据集支持进度描述
            print(f"正在使用标准Dataset类型，启用进度条显示")
            map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
        else:
            print(f"使用IterableDataset类型，无进度条显示")

        dataset = dataset.map(
            maybe_convert_to_chatml,
            remove_columns="conversations" if "conversations" in dataset.column_names else None,
            **map_kwargs,
        )
        print(f"转换完成，当前数据集列名：{dataset.column_names}")

        # --- 应用聊天模板 ---
        print(f"\n=== 阶段2：应用聊天模板 ===")
        if "messages" in dataset.column_names:
            print(f"检测到消息列[messages]，将应用模板")
        else:
            print(f"未检测到消息列，跳过模板应用")

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={"tokenizer": processing_class},
            remove_columns="messages" if "messages" in dataset.column_names else None,
            **map_kwargs,
        )
        print(f"模板应用完成，当前数据集列名：{dataset.column_names}")
        if "text" in dataset.column_names:
            print(f"示例文本内容：\n{dataset[0]['text'][:100]}...")  # 打印首条文本前100字符

        # --- 分词处理 ---
        print(f"\n=== 阶段3：分词处理 ===")
        if not is_processed:
            print(f"数据未预处理，开始分词（处理类：{type(processing_class).__name__}）")
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
            
            def tokenize(example, processing_class, dataset_text_field):
                print(f"正在处理样本ID：{example.get('id', 'N/A')}", " 分词处理的字段名称：", dataset_text_field) if bPrintMoreKv else None
                # print("使用的函数：", processing_class) 
                result = processing_class(example[dataset_text_field])
                if bPrintMoreKv:
                    print(f"分词结果长度：{len(result['input_ids'])}")
                    print(f"示例输入IDs：{result['input_ids'][:10]}...")
                return result
                
            dataset = dataset.map(
                tokenize,
                fn_kwargs={"processing_class": processing_class, "dataset_text_field": args.dataset_text_field},
                **map_kwargs,
            )
            print(f"分词完成，当前数据集列名：{dataset.column_names}")
        else:
            print("数据已预处理，跳过分词步骤")

        # --- 打包/截断处理 ---
        print(f"\n=== 阶段4：序列整理 ===")
        if packing:
            print(f"启用打包模式，最大序列长度：{args.max_seq_length}")
            if args.max_seq_length is None:
                raise ValueError("打包模式必须指定max_seq_length参数！")
            
            print("筛选保留input_ids列...")
            dataset = dataset.select_columns("input_ids")
            
            def pack_examples(examples, seq_length):
                print(f"批量处理样本数：{len(examples['input_ids'])}") if bPrintMoreKv else None
                # ...（实际打包逻辑）
                return packed_examples
                
            dataset = dataset.map(
                pack_examples, 
                batched=True, 
                fn_kwargs={"seq_length": args.max_seq_length}, 
                **map_kwargs
            )
            print(f"打包后数据集结构：{dataset}")
        elif args.max_seq_length is not None:
            print(f"启用截断模式，最大序列长度：{args.max_seq_length}")
            
            def truncate(example, max_seq_length):
                if bPrintMoreKv:
                    print(f"截断前长度：input_ids={len(example['input_ids'])}, attention_mask={len(example['attention_mask'])}")
                truncated = {key: val[:max_seq_length] for key, val in example.items() if key in ["input_ids", "attention_mask"]}
                if bPrintMoreKv:
                    print(f"截断后长度：input_ids={len(truncated['input_ids'])}, attention_mask={len(truncated['attention_mask'])}")
                return truncated
                
            dataset = dataset.map(
                truncate,
                fn_kwargs={"max_seq_length": args.max_seq_length},
                **map_kwargs,
            )
            print(f"截断后示例长度：{len(dataset[0]['input_ids'])}")
        # else:
        #     print("未指定max_seq_length，跳过序列整理")


            # For Liger kernel, ensure only input_ids is present
            if args.use_liger:
                print("补充ids")
                dataset = dataset.select_columns("input_ids")

        # 包含网站下载的字段，和新增的字段
        print("处理完返回的数据：", dataset)
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            accuracy = (correct_tokens.sum() / total_tokens.sum()).item() if total_tokens.sum() > 0 else 0.0
            self._metrics["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
