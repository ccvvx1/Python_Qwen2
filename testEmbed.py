import torch
import torch.nn as nn

# 定义参数
vocab_size = 10  # 假设词表共有50,000个token
embedding_dim = 5  # 目标维度

# 创建嵌入层
embedding_layer = nn.Embedding(
    num_embeddings=vocab_size, 
    embedding_dim=embedding_dim
)

print(embedding_layer)

def print_detail(token_id) :
    input_tensor = torch.tensor([[token_id]])  # 
    print(input_tensor)

    embedded = embedding_layer(input_tensor)
    print(embedded.shape)
    # 输出: torch.Size([1, 1, 600])

    vector_600d = embedded[0, 0]  # 形状: [600]
    print(vector_600d)

print_detail(3)
print_detail(2)
print_detail(3)