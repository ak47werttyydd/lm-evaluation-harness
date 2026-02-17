#!/bin/bash
# Evaluate all GDN-340M checkpoints in hf_ckpt/

CKPT_DIR="/home/a84400789/lm-evaluation-harness/hf_ckpt"
TASKS="arc_easy,arc_challenge,hellaswag,lambada_standard,lambada_openai,piqa,openbookqa"

for model_dir in "$CKPT_DIR"/*/; do
    dirname=$(basename "$model_dir")
    # Extract gbs number: e.g. gbs163840tk.lr3e-4.step122070 -> 163840
    gbs=$(echo "$dirname" | grep -oP '(?<=gbs)\d+(?=tk)')

    if [ -z "$gbs" ]; then
        echo "WARNING: Could not extract gbs from $dirname, skipping"
        continue
    fi

    output_path="./eval_results/gdn_340M_gbs${gbs}tokens"

    echo "========================================="
    echo "Evaluating: $dirname"
    echo "GBS tokens: $gbs"
    echo "Output: $output_path"
    echo "========================================="

    python -m fla_eval --model hf \
        --model_args pretrained="${model_dir}",trust_remote_code=True,dtype=bfloat16 \
        --tasks $TASKS \
        --batch_size auto \
        --output_path "$output_path"

    echo "Done: $dirname"
    echo ""
done

echo "All evaluations complete!"

