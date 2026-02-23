#!/bin/bash

BASE_DIR="/home/a84400789/flame/exp/gdn-340M-4K-20B"
TOKENIZER="m-a-p/340M-20B-GatedDeltaNet-pure-baseline"

# 遍历所有以 hybrid.lr3e-4 开头的文件夹
for exp_dir in "$BASE_DIR"/hybrid.lr3e-4*/; do
    exp_dir="${exp_dir%/}"
    dir_name=$(basename "$exp_dir")

    # 根据文件夹名选择对应的 config
    if [[ "$dir_name" == *transformer* ]]; then
        CONFIG="/home/a84400789/flame/configs/transformer_340M.json"
    elif [[ "$dir_name" =~ hybrid340M([0-9]) ]]; then
        M_NUM="${BASH_REMATCH[1]}"
        CONFIG="/home/a84400789/flame/configs/hybrid340M${M_NUM}.json"
    else
        echo "[SKIP] Unknown config type for: $dir_name"
        continue
    fi

    checkpoint_dir="$exp_dir/checkpoint"

    if [ ! -d "$checkpoint_dir" ]; then
        echo "[SKIP] No checkpoint dir found in: $exp_dir"
        continue
    fi

    # 找到 step 编号最大的 step-* 文件夹
    latest_step=""
    latest_num=-1

    for step_dir in "$checkpoint_dir"/step-*/; do
        step_dir="${step_dir%/}"
        step_name=$(basename "$step_dir")
        step_num="${step_name#step-}"

        if [[ "$step_num" =~ ^[0-9]+$ ]]; then
            if [ "$step_num" -gt "$latest_num" ]; then
                latest_num="$step_num"
                latest_step="$step_num"
            fi
        fi
    done

    if [ -z "$latest_step" ]; then
        echo "[SKIP] No valid step-* dirs found in: $checkpoint_dir"
        continue
    fi

    echo "======================================"
    echo "[RUN] $exp_dir"
    echo "      Latest step: $latest_step"
    echo "      Config: $CONFIG"
    echo "======================================"

    #default largest batch size in H100 is 64
    python -m flame.utils.convert_dcp_to_hf \
        --path "$exp_dir" \
        --step "$latest_step" \
        --config "$CONFIG" \
        --tokenizer "$TOKENIZER" 

    if [ $? -eq 0 ]; then
        echo "[OK] Conversion succeeded for step $latest_step in $dir_name"
    else
        echo "[ERROR] Conversion FAILED for step $latest_step in $dir_name"
    fi

    echo ""
done

echo "All done."