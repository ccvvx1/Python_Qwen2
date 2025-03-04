import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "../Python_Qwen2"
# model_name = "./empty_qwen2"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
# tokenizer.save_pretrained("../Python_Qwen2")
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id
print(model)
# text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is ok good good good-look"
# inputs = tokenizer(text, return_tensors="pt")

# {% if not add_generation_prompt is defined %}        → 设置默认生成提示参数
# {% set ns = namespace(...) %}                        → 初始化状态变量
# {{ bos_token }}{{ ns.system_prompt }}               → 添加起始符和系统提示
# {% for message in messages %}                       → 遍历所有对话消息
#     → 处理系统提示（存储到ns.system_prompt）
#     → 用户消息格式化为"👤+内容"
#     → 助手工具调用显示为JSON代码块
#     → 工具执行结果格式化为"💡+内容"
#     → 常规助手回复显示为"🤖+内容"
# {% endfor %}
# {% if add_generation_prompt %}                       → 添加生成提示符


messages = [
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm fine."},
    {"role": "user", "content": "a simple java example"}
]

formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, 
    add_generation_prompt=True)
# print(formatted_input)

print("输入内容：", formatted_input)
inputs = tokenizer(formatted_input, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens1=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)