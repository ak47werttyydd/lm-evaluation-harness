#!/bin/bash
CKPT_DIR="/home/a84400789/lm-evaluation-harness/hf_ckpt"
RESULTS_DIR="/home/a84400789/lm-evaluation-harness/eval_results"
TASKS="arc_easy,arc_challenge,hellaswag,lambada_standard,lambada_openai,piqa,openbookqa"

MODELS=(
    "gdn-340M-4K-20B.head.lr3e-4.gdn340M_head4_dim256.ngpu8.bs12.ga1.steps50862.ckpt_step50862"
    "gdn-340M-4K-20B.head.lr3e-4.gdn340M_head8_dim128.ngpu8.bs12.ga1.steps50862.ckpt_step50862"
    "gdn-340M-4K-20B.head.lr3e-4.gdn340M_head16_dim64.ngpu8.bs12.ga1.steps50862.ckpt_step50862"
)

for dirname in "${MODELS[@]}"; do
    model_dir="${CKPT_DIR}/${dirname}"
    output_path="${RESULTS_DIR}/${dirname}"

    echo "========================================="
    echo "Evaluating: $dirname"
    echo "Output: $output_path"
    echo "========================================="

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
done

echo "All evaluations complete!"