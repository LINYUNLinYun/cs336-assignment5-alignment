#!/usr/bin/env bash

# ============================================================
# 该脚本和 run_train_grpo.sh 并无区别，只是为了能两个实验一起跑开的

set -euo pipefail

cd /root/code/cs336-assignments/CS336-assignment5-alignment

MODEL="/root/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf"
PROMPT="question_only"
TRAIN_DATA="data/gsm8k/train.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"

SEEDS=(42 43)
LEARNING_RATE=1e-5

# GPU 1 trains the policy; GPU 3 serves vLLM rollouts. Port 8001 avoids the
# existing GPU 0/2 experiment, which uses port 8000.
TRAIN_GPU=1
VLLM_GPU=3
VLLM_PORT=8001
VLLM_GPU_MEMORY_UTILIZATION=0.80

# train_grpo addresses the physical GPU indices above directly.
unset CUDA_VISIBLE_DEVICES

EXPERIMENT_NAME="standard_on_policy_${PROMPT}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="results/${EXPERIMENT_NAME}_${RUN_STAMP}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

WANDB_PROJECT="cs336-assignment5-grpo"
WANDB_ENTITY="sysu-linyun86886"
WANDB_MODE="online"

export WANDB_PROJECT
export WANDB_DIR="${OUTPUT_ROOT}/wandb"
export WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb_cache"

N_TRAIN_EXAMPLES=6400
N_VAL_EXAMPLES=1024
NUM_ROLLOUT_STEPS=200

ROLLOUT_BATCH_SIZE=256
TRAIN_BATCH_SIZE=256
GROUP_SIZE=8
GRADIENT_ACCUMULATION_STEPS=128
MAX_GRAD_NORM=1.0

SAMPLING_TEMPERATURE=1.0
SAMPLING_TOP_P=1.0
SAMPLING_MAX_TOKENS=512
VLLM_REQUEST_BATCH_SIZE=64

EVAL_INTERVAL=10
ROLLOUT_LOG_INTERVAL=40
CHECKPOINT_INTERVAL=0
WANDB_SAMPLE_COUNT=16

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

echo "Question-only GRPO training"
echo "Policy GPU    : ${TRAIN_GPU}"
echo "vLLM GPU      : ${VLLM_GPU}"
echo "vLLM port     : ${VLLM_PORT}"
echo "Learning rate : ${LEARNING_RATE}"
echo "Seeds         : ${SEEDS[*]}"
echo "Output root   : ${OUTPUT_ROOT}"

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="${EXPERIMENT_NAME}-lr${LEARNING_RATE}-seed${SEED}-${RUN_STAMP}"
    RUN_OUTPUT="${OUTPUT_ROOT}/seed_${SEED}"
    RUN_LOG="${LOG_ROOT}/seed_${SEED}.log"

    mkdir -p "${RUN_OUTPUT}"
    echo
    echo "Starting seed ${SEED}; log: ${RUN_LOG}"

    uv run python -m cs336_alignment.train_grpo \
        --model "${MODEL}" \
        --prompt "${PROMPT}" \
        --reward-fn auto \
        --train-data "${TRAIN_DATA}" \
        --val-data "${VAL_DATA}" \
        --output-dir "${RUN_OUTPUT}" \
        --n-train-examples "${N_TRAIN_EXAMPLES}" \
        --n-val-examples "${N_VAL_EXAMPLES}" \
        --num-rollout-steps "${NUM_ROLLOUT_STEPS}" \
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --group-size "${GROUP_SIZE}" \
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --learning-rate "${LEARNING_RATE}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --sampling-temperature "${SAMPLING_TEMPERATURE}" \
        --sampling-top-p "${SAMPLING_TOP_P}" \
        --sampling-max-tokens "${SAMPLING_MAX_TOKENS}" \
        --vllm-request-batch-size "${VLLM_REQUEST_BATCH_SIZE}" \
        --eval-interval "${EVAL_INTERVAL}" \
        --rollout-log-interval "${ROLLOUT_LOG_INTERVAL}" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        --eval-before-training \
        --seed "${SEED}" \
        --train-gpu "${TRAIN_GPU}" \
        --vllm-gpu "${VLLM_GPU}" \
        --vllm-port "${VLLM_PORT}" \
        --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --launch-vllm \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-run-name "${RUN_NAME}" \
        --wandb-mode "${WANDB_MODE}" \
        --wandb-sample-count "${WANDB_SAMPLE_COUNT}" \
        2>&1 | tee "${RUN_LOG}"

    echo "Finished seed ${SEED}"
done

echo
echo "All question-only runs completed: ${OUTPUT_ROOT}"
