from datasets import load_dataset

# # Load the dataset
# # dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", token="YOUR_HF_TOKEN")
# dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B")
# # dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k")
# dataset = dataset["train"]


from datasets import DatasetDict, Dataset

data = {
    "train": [
        {"instruction": "你好吗", "response": "<think>说的是什么</think>你大爷是吧"},
        {"instruction": "不会吧啊", "response": "<think>不太懂</think>你真好"},
        {"instruction": "天气不错啊", "response": "<think>向天</think>天天晒太阳"},
        {"instruction": "配套也是不错的", "response": "<think>啰嗦</think>非常标准的配套"},
        {"instruction": "厂商服务到位是的", "response": "<think>天啊</think>昨天的描述"},
    ]
}

# 构造分片数据集
dataset_dict = DatasetDict({
    "train": Dataset.from_list(data["train"])  # 将列表转换为Dataset对象‌:ml-citation{ref="1,4" data="citationList"}
})

dataset = dataset_dict["train"]  # 正确访问分片
# sub_dataset = dataset.select(range(5))  # 可正常操作

sub_dataset = dataset.select(range(5))  # 假设需要处理train分片

# Format the dataset
def format_instruction(example):
    return {
        "text": (
            "<|user|>\n"
            f"{example['instruction']}\n"
            "<|end|>\n"
            "<|assistant|>\n"
            f"{example['response']}\n"
            "<|end|>"
        )
    }

# formatted_dataset = sub_dataset.map(format_instruction, batched=False, remove_columns=['conversation_id', 'conversations', 'gen_input_configs', 'gen_response_configs', 'intent', 'knowledge', 'difficulty', 'difficulty_generator', 'input_quality', 'quality_explanation', 'quality_generator', 'task_category', 'other_task_category', 'task_category_generator', 'language'])
formatted_dataset = sub_dataset.map(format_instruction, batched=False)

# formatted_dataset = sub_dataset.map(format_instruction, batched=False)
formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)  # 90-10 train-test split

print(formatted_dataset["test"])



# from datasets import Dataset
import pandas as pd

# # 假设您的数据集结构如下
# data = {
#     'instruction': ['给出以下文本的情感分析'],
#     'response': ['这段文本表达了积极的情感'],
#     'text': ['我非常喜欢这个产品的设计！']
# }
# dataset = Dataset.from_dict(data)

# 方法一：转换为Pandas DataFrame打印（推荐）
# print("\n方法一：转换为DataFrame显示")
# df = pd.DataFrame(formatted_dataset["test"])
# print(df.to_string(index=False))  # 禁用行索引

# 方法二：逐行格式化打印
print("\n方法二：结构化遍历输出")
for i in range(len(formatted_dataset["test"])):
    print(f"\nRow {i+1}")
    for feature in formatted_dataset["test"].features:
        value = formatted_dataset["test"][i][feature]
        # 处理长文本的显示
        display_value = str(value)[:50] + "..." if len(str(value)) > 50 else value
        print(f"  {feature.upper():<12} ▶  {display_value}")

# # 方法三：原始结构打印（适合调试）
# print("\n方法三：原始数据结构")
# print(dataset)





from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_id = "distilbert/distilgpt2"
# model_id = "google-bert/bert-base-chinese"
model_id = "../../Python_Qwen2"
# model_id = "microsoft/phi-3-mini-4k-instruct"
print("通过配置构建标签生成需要的函数：")
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# print("token类型：", tokenizer)

from llama.tokenization_llama_fast import LlamaTokenizerFast

print("Q:token.from_pretrained会路由到哪个地方？\nA:会路由到/content/Python_Qwen2/trainQwen2/tokenization_utils_base.py的1800行\nFile \"/content/Python_Qwen2/trainQwen2/tokenization_utils_base.py\", line 1800")
tokenizer = LlamaTokenizerFast.from_pretrained(model_id, trust_remote_code=True)

# Add custom tokens
# CUSTOM_TOKENS = ["", ""]
# tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_TOKENS})
# tokenizer.pad_token = tokenizer.eos_token

# Load model with flash attention
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     use_flash_attention_2=False,  # 禁用 FlashAttention
#     device_map="auto",
#     torch_dtype=torch.float16,
#     attn_implementation="flash_attention_2"
# )
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    is_decoder=True,
    use_flash_attention_2=False,  # 禁用 FlashAttention
    # device_map="auto",
    torch_dtype=torch.float16
)
# model.resize_token_embeddings(len(tokenizer))  # Resize for custom tokens


from speft import LoraConfig

peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.2,  # Dropout rate
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    # target_modules=["query", "key", "value"],
    target_modules=["q_proj", "k_proj", "v_proj"],
    bias="none",  # No bias terms
    task_type="CAUSAL_LM"  # Task type
)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="../../train_qwen2",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=2,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)

from strl import SFTTrainer
from data import DataCollatorForLanguageModeling

print("自定义加载数据：", DataCollatorForLanguageModeling)
print("标签：", tokenizer)
# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    data_collator=data_collator,
    # processing_class=LlamaTokenizerFast,
    peft_config=peft_config
)


trainer.train()
trainer.save_model("../../train_qwen2")
# model.save_pretrained("./phi-3-deepseek-finetuned-final")
tokenizer.save_pretrained("../../train_qwen2")