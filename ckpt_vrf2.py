import fla
from transformers import AutoModelForCausalLM, AutoConfig
import torch

path = "/home/a84400789/lm-evaluation-harness/hf_ckpt/gbs32768tk.lr3e-4.step610351"

config = AutoConfig.from_pretrained(path, trust_remote_code=True)
print(f"Config vocab_size: {config.vocab_size}")
print(f"Config model_type: {config.model_type}")
print(f"Config architectures: {config.architectures}")

# 加载时直接指定 device，或加载后 .cuda()
model = AutoModelForCausalLM.from_pretrained(
    path, 
    trust_remote_code=True, 
    dtype=torch.bfloat16,   # 用 dtype 替代已弃用的 torch_dtype
    device_map="cuda"        # 直接放到 GPU
)

for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        nan_count = torch.isnan(param).sum().item()
        inf_count = torch.isinf(param).sum().item()
        print(f"!!! NaN/Inf in {name}: nan={nan_count}, inf={inf_count}")

# input_ids 也要在同一设备上
input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")

with torch.no_grad():
    output = model(input_ids)
    logits = output.logits
    print(f"Logits shape: {logits.shape}")
    print(f"Logits has NaN: {torch.isnan(logits).any()}")
    print(f"Logits sample: {logits[0, -1, :10]}")
    print(f"Logits std: {logits.std()}")

# 检查 A_log 对应的实际 decay 值
# for name, param in model.named_parameters():
#     if 'A_log' in name:
#         a = param.float()
#         print(f"{name}: NaN={torch.isnan(a).sum()}, "
#               f"exp(A) NaN={torch.isnan(a.exp()).sum()}, "
#               f"exp(A) range=[{a.exp().nanmean():.4f}]")