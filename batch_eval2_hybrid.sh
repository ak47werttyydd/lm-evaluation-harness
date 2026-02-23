#!/bin/bash
# Evaluate all GDN-340M checkpoints in hf_ckpt/

CKPT_DIR="/home/a84400789/lm-evaluation-harness/hf_ckpt"
RESULTS_DIR="/home/a84400789/lm-evaluation-harness/eval_results"
TASKS="arc_easy,arc_challenge,hellaswag,lambada_standard,lambada_openai,piqa,openbookqa"

for model_dir in "$CKPT_DIR"/*/; do
    dirname=$(basename "$model_dir")
    if [[ "$dirname" == *hybrid* ]]; then
        output_path="${RESULTS_DIR}/${dirname}"

        echo "========================================="
        echo "Evaluating: $dirname"
        echo "Output: $output_path"
        echo "========================================="

        # python -m fla_eval --model hf \
        #     --model_args pretrained="${model_dir}",trust_remote_code=True,torch_dtype=bfloat16 \
        #     --tasks $TASKS \
        #     --batch_size auto \
        #     --output_path "$output_path"

        python -m fla_eval --model hf \
            --model_args pretrained="${model_dir}",trust_remote_code=True,dtype=bfloat16 \
            --tasks $TASKS \
            --batch_size auto \
            --output_path "$output_path" \
            --num_fewshot 0 \
            --device cuda \
            --show_config

        echo "Done: $dirname"
        echo ""
    else
        continue
    fi
done

echo "All evaluations complete!"