from transformers import AutoConfig, AutoModelForCausalLM
import torch

# 配置参数
config = {
    "architectures": ["Qwen2ForCausalLM"],
    "vocab_size": 600,
    "hidden_size": 1536,
    "intermediate_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "max_position_embeddings": 600,
    "torch_dtype": "float16"
}

# 创建配置
model_config = AutoConfig.for_model("qwen2", **config)

# 初始化空模型
model = AutoModelForCausalLM.from_config(model_config)

# 验证参数
print("Model Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.mean().item():.4f} (mean) ± {param.data.std().item():.4f} (std)")

# 保存模型
model.save_pretrained("./empty_qwen2")
model_config.save_pretrained("./empty_qwen2")
