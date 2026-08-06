#!/usr/bin/env bash
set -euo pipefail

cd /root/code/cs336-assignments/CS336-assignment5-alignment

# ============================================================
# 固定实验配置
# ============================================================

MODEL="/root/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf"
PROMPT="r1_zero"

TRAIN_DATA="data/gsm8k/train.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"

EXPERIMENT_NAME="learning_rate_sweep_${PROMPT}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="results/${EXPERIMENT_NAME}_${RUN_STAMP}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

# 使用与默认学习率实验相同的随机种子，便于公平比较
SEEDS=(44 45)

# 1e-5 的 seed 44、45 已经跑过，因此这里只跑新的学习率
LEARNING_RATES=(5e-6 2e-5)

# 如果希望重新完整跑一遍三个学习率，改成：
# LEARNING_RATES=(5e-6 1e-5 2e-5)

# ============================================================
# GPU 配置
# ============================================================

unset CUDA_VISIBLE_DEVICES

TRAIN_GPU=0
VLLM_GPU=2
VLLM_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.80

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ============================================================
# W&B 配置
# ============================================================

WANDB_PROJECT="cs336-assignment5-grpo"
WANDB_ENTITY="sysu-linyun86886"
WANDB_MODE="online"

export WANDB_PROJECT
export WANDB_DIR="${OUTPUT_ROOT}/wandb"
export WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb_cache"

# ============================================================
# GRPO 超参数
# 除 learning rate 外，全部保持与标准实验相同
# ============================================================

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

mkdir -p \
    "${OUTPUT_ROOT}" \
    "${LOG_ROOT}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}"

BLUE="\033[1;34m"
RESET="\033[0m"

printf "${BLUE}======================================================================================${RESET}\n"
printf "${BLUE}=============================== LR SWEEP TRAINING START ==============================${RESET}\n"
printf "${BLUE}======================================================================================${RESET}\n"

echo "Model          : ${MODEL}"
echo "Prompt         : ${PROMPT}"
echo "Learning rates : ${LEARNING_RATES[*]}"
echo "Seeds          : ${SEEDS[*]}"
echo "Output root    : ${OUTPUT_ROOT}"
echo "W&B project    : ${WANDB_PROJECT}"
echo "W&B entity     : ${WANDB_ENTITY}"
echo "======================================================================================"

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    LR_OUTPUT_ROOT="${OUTPUT_ROOT}/lr_${LEARNING_RATE}"
    LR_LOG_ROOT="${LOG_ROOT}/lr_${LEARNING_RATE}"

    mkdir -p "${LR_OUTPUT_ROOT}" "${LR_LOG_ROOT}"

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${EXPERIMENT_NAME}-lr${LEARNING_RATE}-seed${SEED}-${RUN_STAMP}"
        RUN_OUTPUT="${LR_OUTPUT_ROOT}/seed_${SEED}"
        RUN_LOG="${LR_LOG_ROOT}/seed_${SEED}.log"

        mkdir -p "${RUN_OUTPUT}"

        echo
        echo "======================================================================================"
        echo "Starting run"
        echo "Learning rate : ${LEARNING_RATE}"
        echo "Seed          : ${SEED}"
        echo "Run name      : ${RUN_NAME}"
        echo "Output        : ${RUN_OUTPUT}"
        echo "Log           : ${RUN_LOG}"
        echo "======================================================================================"

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

        echo "Finished: learning_rate=${LEARNING_RATE}, seed=${SEED}"
    done
done

echo
echo "======================================================================================"
echo "All learning-rate sweep runs completed"
echo "Results: ${OUTPUT_ROOT}"
echo "Logs   : ${LOG_ROOT}"
echo "======================================================================================"