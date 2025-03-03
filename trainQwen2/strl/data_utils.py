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

from typing import Any, Callable, Optional, Sequence, TypeVar, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)


def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False


def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    添加了详细处理过程追踪的模板应用函数
    """
    print("\n=== 开始应用聊天模板 ===")
    print(f"输入数据键: {example.keys()}")
    print(f"工具参数类型: {type(tools) if tools else '无工具参数'}")

    # 验证输入键有效性
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    valid_key_combinations = [
        {"messages"}, {"prompt"}, {"prompt", "completion"}, 
        {"prompt", "chosen", "rejected"}, {"chosen", "rejected"},
        {"prompt", "completion", "label"}
    ]
    
    print(f"\n[键验证] 检测到有效键组合: {example_keys}")
    if example_keys not in valid_key_combinations:
        print(f"❌ 无效键组合! 允许的组合: {valid_key_combinations}")
        raise KeyError(f"Invalid keys in the example: {example_keys}")
    else:
        print("✅ 键组合验证通过")

    result = {}
    
    # 处理messages类型
    if "messages" in example:
        print("\n[消息处理] 检测到messages键")
        print(f"消息数量: {len(example['messages'])}")
        print("首条消息结构:", example["messages"][0] if example["messages"] else "空")
        
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)
        result["text"] = messages
        print(f"模板应用结果预览: {messages[:100]}...")

    # 处理prompt类型
    if "prompt" in example:
        print("\n[提示处理] 检测到prompt键")
        last_role = example["prompt"][-1]["role"]
        print(f"最后消息角色: {last_role}")
        
        add_generation_prompt = last_role == "user"
        continue_final_message = last_role == "assistant"
        print(f"生成提示标记: {add_generation_prompt} | 继续标记: {continue_final_message}")
        
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        result["prompt"] = prompt
        print(f"提示模板预览: {prompt[:100]}...")
# def ok():
    print("\n=== 开始处理数据样本 ===")
    print(f"输入数据包含的键: {example.keys()}")
    
    # 显式prompt处理分支
    if "prompt" in example:
        print("\n[显式Prompt处理]")
        print(f"原始prompt消息数: {len(example['prompt'])}")
        print(f"原始prompt最后一条消息: {example['prompt'][-1]}")

        # 处理chosen分支
        if "chosen" in example:
            print(f"\n处理chosen分支，消息数: {len(example['chosen'])}")
            combined_chosen = example["prompt"] + example["chosen"]
            print(f"拼接后总消息数: {len(combined_chosen)}")
            prompt_chosen = tokenizer.apply_chat_template(combined_chosen, tools=tools, tokenize=False)
            print(f"完整chosen文本长度: {len(prompt_chosen)}")
            chosen = prompt_chosen[len(prompt):]
            print(f"截取后chosen长度: {len(chosen)} | 截取部分示例: {chosen[:50]}...")

        # 处理rejected分支
        if "rejected" in example:
            print(f"\n处理rejected分支，消息数: {len(example['rejected'])}")
            combined_rejected = example["prompt"] + example["rejected"]
            prompt_rejected = tokenizer.apply_chat_template(combined_rejected, tools=tools, tokenize=False)
            print(f"完整rejected文本长度: {len(prompt_rejected)}")
            rejected = prompt_rejected[len(prompt):]
            print(f"截取后rejected长度: {len(rejected)} | 截取部分示例: {rejected[:50]}...")

        # 处理completion分支
        if "completion" in example:
            print(f"\n处理completion分支，消息数: {len(example['completion'])}")
            combined_completion = example["prompt"] + example["completion"]
            prompt_completion = tokenizer.apply_chat_template(combined_completion, tools=tools, tokenize=False)
            print(f"完整completion文本长度: {len(prompt_completion)}")
            completion = prompt_completion[len(prompt):]
            print(f"截取后completion长度: {len(completion)} | 截取部分示例: {completion[:50]}...")

    else:
        print("\n[隐式Prompt处理]")
        # 处理chosen/rejected分支
        if "chosen" in example:
            print(f"直接处理chosen消息数: {len(example['chosen'])}")
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
            print(f"chosen最终文本长度: {len(chosen)} | 示例: {chosen[:50]}...")

        if "rejected" in example:
            print(f"直接处理rejected消息数: {len(example['rejected'])}")
            rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)
            print(f"rejected最终文本长度: {len(rejected)} | 示例: {rejected[:50]}...")

    # 验证prompt一致性
    if "prompt" in example:
        print("\n[一致性验证]")
        error_template = "检测到潜在不一致:\nPrompt:\n%s\n\n组合文本:\n%s"
        
        def check_consistency(full_text, name):
            if not full_text.startswith(prompt):
                print("❌ 验证失败: %s文本不以prompt开头" % name)
                print(error_template % (prompt[:100], full_text[:100]))
                return False
            print("✅ %s验证通过" % name)
            return True

        if "chosen" in example:
            check_consistency(prompt_chosen, "chosen")
        if "rejected" in example:
            check_consistency(prompt_rejected, "rejected")
        if "completion" in example:
            check_consistency(prompt_completion, "completion")

    # 构建输出结果
    print("\n[构建输出结果]")
    output = {}
    key_mapping = {
        "messages": ("text", messages),
        "prompt": ("prompt", prompt),
        "chosen": ("chosen", chosen),
        "rejected": ("rejected", rejected),
        "completion": ("completion", completion),
        "label": ("label", example.get("label"))
    }

    for key in example:
        if key in key_mapping:
            output_key, value = key_mapping[key]
            output[output_key] = value
            status = "存在" if value is not None else "缺失"
            print(f"添加字段: {output_key.ljust(8)} | 状态: {status.ljust(4)} | 长度: {len(value) if isinstance(value, str) else 'N/A'}")

    print("\n=== 最终输出 ===")
    print({k: (f"{v[:50]}..." if isinstance(v, str) else v) for k, v in output.items()})
    return output



def maybe_apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    If the example is in a conversational format, apply a chat template to it.

    Args:
        example (`dict[str, list[dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset type. The supported dataset types are:

                - Language modeling dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer to apply the chat template with.
        tools (`list[Union[dict, Callable]]` or `None`, *optional*, defaults to `None`):
            A list of tools (callable functions) that will be accessible to the model.
            If the template does not support function calling, this argument will have no effect

    Returns:
        `dict[str, str]`:
            Formatted example with the chat template applied.

    Notes:
        - This function does not alter the keys, except for Language modeling dataset, where `"messages"` is replaced
        by `"text"`.

        - In case of prompt-only data, if the last role is `"user"`, the generation prompt is added to the prompt.
        Else, if the last role is `"assistant"`, the final message is continued.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    >>> example = {
    ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    ...     "completion": [{"role": "assistant", "content": "It is blue."}]
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
    ```
    """
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
    else:
        print("直接返回模板数据")
        return example


def _unpair_row(examples: list[dict[str, list[dict[str, str]]]]) -> list[dict[str, list[dict[str, str]]]]:
    batch_size = len(examples["chosen"])
    new_rows = {
        "completion": examples["chosen"] + examples["rejected"],
        "label": [True] * batch_size + [False] * batch_size,
    }
    if "prompt" in examples:
        new_rows["prompt"] = examples["prompt"] + examples["prompt"]
    return new_rows


def unpair_preference_dataset(
    dataset: DatasetType, num_proc: Optional[int] = None, desc: Optional[str] = None
) -> DatasetType:
    r"""
    Unpair a preference dataset.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        desc (`str` or `None`, *optional*, defaults to `None`):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset`: The unpaired preference dataset.

    Example:

    ```python
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"]
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."]
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })
    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """
    return dataset.map(_unpair_row, batched=True, remove_columns=["chosen", "rejected"], num_proc=num_proc, desc=desc)


def maybe_unpair_preference_dataset(
    dataset: DatasetType, num_proc: Optional[int] = None, desc: Optional[str] = None
) -> DatasetType:
    r"""
    Unpair a preference dataset if it is paired.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        desc (`str` or `None`, *optional*, defaults to `None`):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset` or `DatasetDict`: The unpaired preference dataset if it was paired, otherwise the original dataset.

    Example:

    ```python
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"]
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."]
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })
    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """
    if isinstance(dataset, DatasetDict):
        column_names = dataset[list(dataset.keys())[0]].column_names
    else:
        column_names = dataset.column_names
    if "chosen" in column_names and "rejected" in column_names:
        return unpair_preference_dataset(dataset, num_proc=num_proc, desc=desc)
    else:
        return dataset


def extract_prompt(example: dict[str, Sequence]) -> dict[str, Sequence]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both
    the chosen and rejected completions.

    For more details, see [`maybe_extract_prompt`].
    """
    for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
        if example["chosen"][idx] != example["rejected"][idx]:
            if example["chosen"][idx - 1] == " ":  # remove space before the prompt
                idx -= 1
            break
    return {
        "prompt": example["chosen"][:idx],
        "chosen": example["chosen"][idx:],
        "rejected": example["rejected"][idx:],
    }


def maybe_extract_prompt(example: dict[str, list]) -> dict[str, list]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both
    the chosen and rejected completions.

    If the example already contains a `"prompt"` key, the function returns the example as is. Else, the function
    identifies the longest common sequence (prefix) of conversation turns between the "chosen" and "rejected"
    completions and extracts this as the prompt. It then removes this prompt from the respective "chosen" and
    "rejected" completions.

    Args:
        example (`dict[str, list]`):
            A dictionary representing a single data entry in the preference dataset. It must contain the keys
            `"chosen"` and `"rejected"`, where each value is either conversational or standard (`str`).

    Returns:
        `dict[str, list]`: A dictionary containing:
            - `"prompt"`: The longest common prefix between the "chosen" and "rejected" completions.
            - `"chosen"`: The remainder of the "chosen" completion, with the prompt removed.
            - `"rejected"`: The remainder of the "rejected" completion, with the prompt removed.

    Examples:

    ```python
    >>> example = {
    ...     "chosen": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is blue."}
    ...     ],
    ...     "rejected": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is green."}
    ...     ]
    ... }
    >>> extract_prompt(example)
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```

    Or, with the `map` method of `datasets.Dataset`:

    ```python
    >>> from trl import extract_prompt
    >>> from datasets import Dataset
    >>> dataset_dict = {
    ...     "chosen": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is blue."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sky."},
    ...         ],
    ...     ],
    ...     "rejected": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is green."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sea."},
    ...         ],
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = dataset.map(extract_prompt)
    >>> dataset[0]
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```
    """
    # Some dataset add a `"prompt"` column, even though the prompt is implicit and included in the "chosen" and
    # "rejected" completions. E.g.:
    # {"prompt": "What color is the sky?",
    #  "chosen": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
    #  "rejected": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}]}
    # That's why we check if the prompt is also conversational before deciding not to extract it.
    if "chosen" not in example or "rejected" not in example:  # not a preference example
        return example
    if "prompt" in example:
        # Both conversational or both non-conversational
        chosen_conv = is_conversational({"chosen": example["chosen"]})
        prompt_conv = is_conversational({"prompt": example["prompt"]})
        if (chosen_conv and prompt_conv) or (not chosen_conv and not prompt_conv):
            return example
    return extract_prompt({"chosen": example["chosen"], "rejected": example["rejected"]})


def pack_examples(examples: dict[str, list[list]], seq_length: int) -> dict[str, list[list]]:
    """
    Pack examples into chunks of size `seq_length`.

    Args:
        examples (`dict[str, list[list]]`):
            Dictionary of examples with keys as strings and values as lists of lists.
        seq_length (`int`):
            Maximum sequence length.

    Returns:
        `dict[str, list[list]]`: Dictionary of examples with keys as strings and values as lists of lists.

    Example:

    ```python
    >>> from trl import pack_examples
    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> pack_examples(examples, seq_length=5)
    {'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8]], 'attention_mask': [[0, 1, 1, 0, 0], [1, 1, 1]]}
    >>> pack_examples(examples, seq_length=2)
    {'input_ids': [[1, 2], [3, 4], [5, 6], [7, 8]], 'attention_mask': [[0, 1], [1, 0], [0, 1], [1, 1]]}
    ```
    """
    # Join  all the values into a single list
    examples = {k: sum(v, []) for k, v in examples.items()}
    # Split the values into chunks of size seq_length
    examples = {k: [v[i : i + seq_length] for i in range(0, len(v), seq_length)] for k, v in examples.items()}
    return examples


def maybe_convert_to_chatml(example: dict[str, list]) -> dict[str, list]:
    """
    Convert a conversational dataset with fields `from` and `value` to ChatML format.
    添加了详细的转换过程打印信息
    """
    print("\n=== 开始ChatML格式转换 ===")
    original_keys = set(example.keys())
    print(f"输入数据原始键: {original_keys}")

    # 遍历所有可能的对话键
    processed_keys = set()
    for key in ["prompt", "completion", "chosen", "rejected", "messages", "conversations"]:
        if key in example and isinstance(example[key], list):
            print(f"\n检测到对话键 [{key}]，包含 {len(example[key])} 条消息")
            messages = example[key]
            modified_count = 0
            
            for i, message in enumerate(messages):
                print(f"\n处理第 {i+1} 条消息:")
                if not isinstance(message, dict):
                    print(f"  警告：消息类型为 {type(message)}，跳过转换")
                    continue
                
                change_flag = False
                # 转换 from -> role
                if "from" in message:
                    original_role = message["from"]
                    message["role"] = message.pop("from")
                    print(f"  ✓ 转换字段: from -> role | 值: {original_role} -> {message['role']}")
                    change_flag = True
                else:
                    print("  × 未找到 'from' 字段")
                
                # 转换 value -> content
                if "value" in message:
                    content_preview = str(message["value"])[:30] + ("..." if len(str(message["value"])) > 30 else "")
                    message["content"] = message.pop("value")
                    print(f"  ✓ 转换字段: value -> content | 内容预览: {content_preview}")
                    change_flag = True
                else:
                    print("  × 未找到 'value' 字段")
                
                if change_flag:
                    modified_count += 1
                    print(f"  当前消息转换后: { {k: str(v)[:50] + ('...' if len(str(v)) > 50 else '') for k, v in message.items()} }")
            
            print(f"共处理 {len(messages)} 条消息，成功转换 {modified_count} 条")
            processed_keys.add(key)

    # 重命名 conversations -> messages
    if "conversations" in example:
        print(f"\n重命名键: conversations -> messages")
        example["messages"] = example.pop("conversations")
        processed_keys.add("conversations")
    
    new_keys = set(example.keys())
    added_keys = new_keys - original_keys
    removed_keys = original_keys - new_keys
    
    print("\n=== 转换结果 ===")
    print(f"新增键: {added_keys if added_keys else '无'}")
    print(f"移除键: {removed_keys if removed_keys else '无'}")
    print(f"最终数据键: {new_keys}")
    print(f"处理过的原始键: {processed_keys if processed_keys else '无'}")
    
    return example

