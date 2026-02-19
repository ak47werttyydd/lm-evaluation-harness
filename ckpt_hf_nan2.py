import fla
from transformers import AutoModelForCausalLM, AutoConfig
import torch
from pathlib import Path
dir=Path("/home/a84400789/lm-evaluation-harness/hf_ckpt")
ckpt_paths = [item for item in dir.iterdir()]
for path in ckpt_paths:
    print(f"Ckpt is at {path}")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    print(f"Config vocab_size: {config.vocab_size}")
    print(f"Config model_type: {config.model_type}")
    print(f"Config architectures: {config.architectures}")

    # 加载时直接指定 device，或加载后 .cuda()
    print("Load checkpoint with bf16")
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,   # transformer < 5.0 must use torch_dtype instead fo dtype
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

    # print("Load checkpoint with f32")
    # model = AutoModelForCausalLM.from_pretrained(
    #     path,
    #     trust_remote_code=True,
    #     dtype=torch.float32,  # 不转 bf16
    #     device_map="cuda"
    # )
    # # for name, param in model.named_parameters():
    # #     if 'A_log' in name:
    # #         print(f"{name}: {param.data.cpu()}")
    # #         print(f"  NaN(f32): {torch.isnan(param).any()}")
    # #         print(f"  NaN(bf16): {torch.isnan(param.to(torch.bfloat16)).any()}")
    # for name, param in model.named_parameters():
    #     if torch.isnan(param).any() or torch.isinf(param).any():
    #         nan_count = torch.isnan(param).sum().item()
    #         inf_count = torch.isinf(param).sum().item()
    #         print(f"!!! NaN/Inf in {name}: nan={nan_count}, inf={inf_count}")
    # # input_ids 也要在同一设备上
    # input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    # with torch.no_grad():
    #     output = model(input_ids)
    #     logits = output.logits
    #     print(f"Logits shape: {logits.shape}")
    #     print(f"Logits has NaN: {torch.isnan(logits).any()}")
    #     print(f"Logits sample: {logits[0, -1, :10]}")
    #     print(f"Logits std: {logits.std()}")

    print(" =============== Next Checkpoint =====================")