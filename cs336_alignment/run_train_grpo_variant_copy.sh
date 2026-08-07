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

EXPERIMENT_NAME="on_policy_variants_gpu23_${PROMPT}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="results/${EXPERIMENT_NAME}_${RUN_STAMP}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

SEEDS=(42 43 44 45)

# GPU 2、3 负责这两个变体
VARIANTS=(
    "rft"
    "maxrl"
)

LEARNING_RATE=1e-5

# ============================================================
# GPU 配置
# ============================================================

unset CUDA_VISIBLE_DEVICES

TRAIN_GPU=2
VLLM_GPU=3
VLLM_PORT=8001
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
# ============================================================

N_TRAIN_EXAMPLES=6400
N_VAL_EXAMPLES=1024
NUM_ROLLOUT_STEPS=200

ROLLOUT_BATCH_SIZE=256
TRAIN_BATCH_SIZE=256
GROUP_SIZE=8
GRADIENT_ACCUMULATION_STEPS=128

ADVANTAGE_EPS=1e-6
MAX_GRAD_NORM=1.0

SAMPLING_TEMPERATURE=1.0
SAMPLING_TOP_P=1.0
SAMPLING_MAX_TOKENS=512
VLLM_REQUEST_BATCH_SIZE=64

NORMALIZATION_CONSTANT=$((TRAIN_BATCH_SIZE * SAMPLING_MAX_TOKENS))

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
YELLOW="\033[1;33m"
RESET="\033[0m"

printf "${BLUE}======================================================================================${RESET}\n"
printf "${BLUE}======================= ON-POLICY VARIANTS: GPU 2 + GPU 3 ============================${RESET}\n"
printf "${BLUE}======================================================================================${RESET}\n"

echo "Model                  : ${MODEL}"
echo "Prompt                 : ${PROMPT}"
echo "Variants               : ${VARIANTS[*]}"
echo "Seeds                  : ${SEEDS[*]}"
echo "Train GPU              : ${TRAIN_GPU}"
echo "vLLM GPU               : ${VLLM_GPU}"
echo "vLLM port              : ${VLLM_PORT}"
echo "Learning rate          : ${LEARNING_RATE}"
echo "Normalization constant : ${NORMALIZATION_CONSTANT}"
echo "Output root            : ${OUTPUT_ROOT}"
echo "======================================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    case "${VARIANT}" in
        rft)
            BASELINE="none"
            ADVANTAGE_NORMALIZER="none"
            LOSS_NORMALIZATION="constant"
            ;;
        maxrl)
            BASELINE="mean"
            ADVANTAGE_NORMALIZER="mean"
            LOSS_NORMALIZATION="constant"
            ;;
        *)
            echo "Unknown variant: ${VARIANT}"
            exit 1
            ;;
    esac

    VARIANT_OUTPUT_ROOT="${OUTPUT_ROOT}/${VARIANT}"
    VARIANT_LOG_ROOT="${LOG_ROOT}/${VARIANT}"

    mkdir -p "${VARIANT_OUTPUT_ROOT}" "${VARIANT_LOG_ROOT}"

    printf "\n${YELLOW}======================================================================================${RESET}\n"
    printf "${YELLOW}Variant: ${VARIANT}${RESET}\n"
    echo "baseline             : ${BASELINE}"
    echo "advantage normalizer : ${ADVANTAGE_NORMALIZER}"
    echo "loss normalization   : ${LOSS_NORMALIZATION}"
    printf "${YELLOW}======================================================================================${RESET}\n"

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${EXPERIMENT_NAME}-${VARIANT}-seed${SEED}-${RUN_STAMP}"
        RUN_OUTPUT="${VARIANT_OUTPUT_ROOT}/seed_${SEED}"
        RUN_LOG="${VARIANT_LOG_ROOT}/seed_${SEED}.log"
        DONE_FILE="${RUN_OUTPUT}/.done"

        mkdir -p "${RUN_OUTPUT}"

        if [[ -f "${DONE_FILE}" ]]; then
            echo "Skipping completed run: variant=${VARIANT}, seed=${SEED}"
            continue
        fi

        echo
        echo "======================================================================================"
        echo "Starting run"
        echo "Variant               : ${VARIANT}"
        echo "Seed                  : ${SEED}"
        echo "Train GPU             : ${TRAIN_GPU}"
        echo "vLLM GPU              : ${VLLM_GPU}"
        echo "Baseline              : ${BASELINE}"
        echo "Advantage normalizer  : ${ADVANTAGE_NORMALIZER}"
        echo "Loss normalization    : ${LOSS_NORMALIZATION}"
        echo "Normalization constant: ${NORMALIZATION_CONSTANT}"
        echo "Run name              : ${RUN_NAME}"
        echo "Output                : ${RUN_OUTPUT}"
        echo "Log                   : ${RUN_LOG}"
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
            --baseline "${BASELINE}" \
            --advantage-normalizer "${ADVANTAGE_NORMALIZER}" \
            --advantage-eps "${ADVANTAGE_EPS}" \
            --importance-reweighting-method "none" \
            --loss-normalization "${LOSS_NORMALIZATION}" \
            --normalization-constant "${NORMALIZATION_CONSTANT}" \
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

        touch "${DONE_FILE}"
        echo "Finished: variant=${VARIANT}, seed=${SEED}"
    done
done

echo
echo "======================================================================================"
echo "GPU 2 + GPU 3 experiments completed"
echo "Results: ${OUTPUT_ROOT}"
echo "Logs   : ${LOG_ROOT}"
echo "======================================================================================"