from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


PROMPT_FILES = {
    "question_only": "cs336_alignment/prompts/question_only.prompt",
    "r1_zero": "cs336_alignment/prompts/r1_zero.prompt",
    "r1_zero_three_shot": "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard on-policy GRPO on GSM8K")

    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--prompt", default="r1_zero")
    parser.add_argument(
        "--reward-fn",
        choices=["auto", "r1_zero", "question_only"],
        default="auto",
    )
    parser.add_argument("--train-data", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val-data", default="data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", default="outputs/grpo_standard_on_policy")

    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-top-p", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--vllm-request-batch-size", type=int, default=64)

    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--rollout-log-interval", type=int, default=40)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument(
        "--eval-before-training",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--launch-vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--wandb-project", default="cs336-assignment5-grpo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-sample-count", type=int, default=16)
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["mean", "none"],
        default="mean",
    )

    parser.add_argument(
        "--advantage-normalizer",
        type=str,
        choices=["std", "none", "mean"],
        default="std",
    )

    parser.add_argument(
        "--advantage-eps",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--importance-reweighting-method",
        type=str,
        choices=["none", "noclip", "grpo", "gspo"],
        default="none",
    )

    parser.add_argument(
        "--loss-normalization",
        type=str,
        choices=["sequence", "constant"],
        default="sequence",
    )

    parser.add_argument(
        "--normalization-constant",
        type=int,
        default=None,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")
    if args.train_gpu == args.vllm_gpu:
        raise ValueError("Training and vLLM must use different GPUs.")
    if max(args.train_gpu, args.vllm_gpu) >= torch.cuda.device_count():
        raise ValueError(
            f"Requested GPUs {args.train_gpu} and {args.vllm_gpu}, "
            f"but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    if args.rollout_batch_size != args.train_batch_size:
        raise ValueError(
            "Standard on-policy GRPO uses every fresh rollout exactly once, so "
            "rollout_batch_size must equal train_batch_size."
        )
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "train_batch_size must be divisible by gradient_accumulation_steps "
            "for the current grpo_train_step implementation."
        )
    for name in (
        "n_train_examples",
        "n_val_examples",
        "num_rollout_steps",
        "rollout_batch_size",
        "group_size",
        "gradient_accumulation_steps",
        "sampling_max_tokens",
        "vllm_request_batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gsm8k(path: str | Path) -> list[dict[str, str]]:
    examples = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "question" not in item or "answer" not in item:
                raise KeyError(f"Missing question/answer at {path}:{line_number}")
            examples.append(
                {
                    "question": item["question"],
                    "answer": item["answer"].rsplit("####", 1)[-1].strip(),
                }
            )
    return examples


def resolve_prompt_and_reward(
    prompt_spec: str,
    reward_fn_name: str,
) -> tuple[str, str, Callable[[str, str], dict[str, float]], bool]:
    if prompt_spec in PROMPT_FILES:
        prompt_name = prompt_spec
        prompt_path = Path(PROMPT_FILES[prompt_spec])
    else:
        prompt_path = Path(prompt_spec)
        prompt_name = prompt_path.stem

    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")

    if reward_fn_name == "auto":
        resolved_reward = (
            "question_only" if "question_only" in prompt_name else "r1_zero"
        )
    else:
        resolved_reward = reward_fn_name

    if resolved_reward == "question_only":
        reward_fn = question_only_reward_fn
        use_answer_stop = False
    else:
        reward_fn = r1_zero_reward_fn
        use_answer_stop = True

    return prompt_name, template, reward_fn, use_answer_stop


def build_prompts(template: str, examples: list[dict[str, str]]) -> list[str]:
    return [template.format(question=example["question"]) for example in examples]


class ShuffledBatchStream:
    def __init__(self, examples: list[dict[str, str]], batch_size: int, seed: int):
        if not examples:
            raise ValueError("Training examples cannot be empty.")
        self.examples = examples
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.order = list(range(len(examples)))
        self.rng.shuffle(self.order)
        self.position = 0

    def next_batch(self) -> list[dict[str, str]]:
        indices = []
        while len(indices) < self.batch_size:
            if self.position == len(self.order):
                self.rng.shuffle(self.order)
                self.position = 0
            take = min(self.batch_size - len(indices), len(self.order) - self.position)
            indices.extend(self.order[self.position : self.position + take])
            self.position += take
        return [self.examples[index] for index in indices]


def make_sampling_params(
    args: argparse.Namespace,
    n: int,
    seed: int,
    use_answer_stop: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n": n,
        "seed": seed,
        "temperature": args.sampling_temperature,
        "top_p": args.sampling_top_p,
        "max_tokens": args.sampling_max_tokens,
    }
    if use_answer_stop:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def generate_rollouts(
    server: VLLMServer,
    prompts: list[str],
    args: argparse.Namespace,
    n: int,
    seed: int,
    use_answer_stop: bool,
) -> list[VLLMCompletion]:
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=make_sampling_params(args, n, seed, use_answer_stop),
        batch_size=args.vllm_request_batch_size,
    )
    expected = len(prompts) * n
    if len(completions) != expected:
        raise RuntimeError(
            f"vLLM returned {len(completions)} completions; expected {expected}."
        )
    return completions


def response_length(completion: VLLMCompletion, tokenizer) -> int:
    if completion.token_ids:
        return len(completion.token_ids)
    return len(tokenizer(completion.text, add_special_tokens=False)["input_ids"])


def make_generation_records(
    examples: list[dict[str, str]],
    prompts: list[str],
    completions: list[VLLMCompletion],
    reward_fn: Callable[[str, str], dict[str, float]],
    tokenizer,
    group_size: int,
) -> list[dict[str, Any]]:
    records = []
    for example_index, (example, prompt) in enumerate(zip(examples, prompts, strict=True)):
        for group_index in range(group_size):
            completion_index = example_index * group_size + group_index
            completion = completions[completion_index]
            scores = reward_fn(completion.text, example["answer"])
            records.append(
                {
                    "example_index": example_index,
                    "group_index": group_index,
                    "question": example["question"],
                    "prompt": prompt,
                    "ground_truth": example["answer"],
                    "response": completion.text,
                    "reward": float(scores["reward"]),
                    "format_reward": float(scores["format_reward"]),
                    "answer_reward": float(scores["answer_reward"]),
                    "response_tokens": response_length(completion, tokenizer),
                    "finish_reason": completion.finish_reason,
                }
            )
    return records


def evaluate_policy(
    server: VLLMServer,
    examples: list[dict[str, str]],
    template: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    tokenizer,
    args: argparse.Namespace,
    use_answer_stop: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    prompts = build_prompts(template, examples)
    completions = generate_rollouts(
        server=server,
        prompts=prompts,
        args=args,
        n=1,
        seed=args.seed + 100_000,
        use_answer_stop=use_answer_stop,
    )
    records = make_generation_records(
        examples=examples,
        prompts=prompts,
        completions=completions,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        group_size=1,
    )

    count = len(records)
    metrics = {
        "val/reward": sum(x["reward"] for x in records) / count,
        "val/format_reward": sum(x["format_reward"] for x in records) / count,
        "val/answer_reward": sum(x["answer_reward"] for x in records) / count,
        "val/avg_response_tokens": sum(x["response_tokens"] for x in records) / count,
        "val/finish_stop_rate": sum(x["finish_reason"] == "stop" for x in records) / count,
        "val/finish_length_rate": sum(x["finish_reason"] == "length" for x in records) / count,
    }
    return metrics, records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def wandb_table(wandb_module, records: list[dict[str, Any]], limit: int):
    columns = [
        "question",
        "ground_truth",
        "response",
        "reward",
        "format_reward",
        "answer_reward",
        "response_tokens",
        "finish_reason",
    ]
    data = [[record[column] for column in columns] for record in records[:limit]]
    return wandb_module.Table(columns=columns, data=data)


def scalar_train_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    mean_reward = metadata.get("mean_rewards", metadata.get("mean_reward", float("nan")))
    return {
        "train/loss": float(metadata["loss"]),
        "train/grad_norm": float(metadata["grad_norm"]),
        "train/token_entropy": float(metadata["token_entropy"]),
        "train/reward": float(mean_reward),
        "train/format_reward": float(metadata["mean_format_reward"]),
        "train/answer_reward": float(metadata["mean_answer_reward"]),
    }


def save_checkpoint(model, tokenizer, output_dir: Path, name: str) -> None:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    torch.cuda.set_device(args.train_gpu)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", vars(args))

    prompt_name, prompt_template, reward_fn, use_answer_stop = resolve_prompt_and_reward(
        args.prompt,
        args.reward_fn,
    )

    train_examples = load_gsm8k(args.train_data)
    val_examples = load_gsm8k(args.val_data)
    if args.n_train_examples > len(train_examples):
        raise ValueError(
            f"Requested {args.n_train_examples} training examples, but found {len(train_examples)}."
        )
    if args.n_val_examples > len(val_examples):
        raise ValueError(
            f"Requested {args.n_val_examples} validation examples, but found {len(val_examples)}."
        )

    subset_rng = random.Random(args.seed)
    subset_rng.shuffle(train_examples)
    train_examples = train_examples[: args.n_train_examples]
    val_examples = val_examples[: args.n_val_examples]

    prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    train_stream = ShuffledBatchStream(
        train_examples,
        batch_size=prompts_per_rollout_batch,
        seed=args.seed + 1,
    )

    wandb_module = None
    wandb_run = None
    try:
        import wandb

        wandb_module = wandb
        run_name = args.wandb_run_name or f"{prompt_name}-seed{args.seed}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config={**vars(args), "resolved_prompt_name": prompt_name},
        )
    except ImportError:
        if args.wandb_mode != "disabled":
            raise

    policy_device = f"cuda:{args.train_gpu}"
    server = VLLMServer(
        model_id=args.model,
        port=args.vllm_port,
        gpu=args.vllm_gpu,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        launch_server=args.launch_vllm,
    )

    completed = False
    try:
        print(f"Starting vLLM on GPU {args.vllm_gpu}...")
        server.start()

        print(f"Loading policy on {policy_device}...")
        policy, tokenizer = get_model_and_tokenizer(args.model, policy_device)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            tokenizer.pad_token = tokenizer.eos_token
        policy.config.use_cache = False
        policy.train()

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        optimizer.zero_grad(set_to_none=True)

        print("Initializing NCCL weight synchronization...")
        server.init_weight_sync(policy_device)
        server.sync_policy_weights(policy)

        metrics_path = output_dir / "metrics.jsonl"
        generations_dir = output_dir / "generations"

        if args.eval_before_training:
            print("Evaluating the initial policy...")
            eval_start = time.perf_counter()
            val_metrics, val_records = evaluate_policy(
                server,
                val_examples,
                prompt_template,
                reward_fn,
                tokenizer,
                args,
                use_answer_stop,
            )
            val_metrics["time/eval_seconds"] = time.perf_counter() - eval_start
            write_jsonl(generations_dir / "val_step_0000.jsonl", val_records)
            step_zero_log: dict[str, Any] = {"step": 0, **val_metrics}
            if wandb_module is not None:
                step_zero_log["val/samples"] = wandb_table(
                    wandb_module,
                    val_records,
                    args.wandb_sample_count,
                )
            if wandb_run is not None:
                wandb_run.log(step_zero_log, step=0)
            append_jsonl(
                metrics_path,
                {key: value for key, value in step_zero_log.items() if isinstance(value, (int, float))},
            )
            print(
                f"step=0 val_reward={val_metrics['val/reward']:.4f} "
                f"val_format={val_metrics['val/format_reward']:.4f}"
            )

        for step in range(1, args.num_rollout_steps + 1):
            step_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(args.train_gpu)

            batch_examples = train_stream.next_batch()
            batch_prompts = build_prompts(prompt_template, batch_examples)
            repeated_prompts = [
                prompt
                for prompt in batch_prompts
                for _ in range(args.group_size)
            ]
            repeated_ground_truths = [
                example["answer"]
                for example in batch_examples
                for _ in range(args.group_size)
            ]

            sync_start = time.perf_counter()
            server.sync_policy_weights(policy)
            sync_seconds = time.perf_counter() - sync_start

            rollout_start = time.perf_counter()
            completions = generate_rollouts(
                server=server,
                prompts=batch_prompts,
                args=args,
                n=args.group_size,
                seed=args.seed + step,
                use_answer_stop=use_answer_stop,
            )
            rollout_seconds = time.perf_counter() - rollout_start
            rollout_responses = [completion.text for completion in completions]

            policy.train()
            train_start = time.perf_counter()
            _, train_metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size,
                baseline=args.baseline,
                advantage_eps=args.advantage_eps,
                advantage_normalizer=args.advantage_normalizer,
                importance_reweighting_method=args.importance_reweighting_method,
                old_log_probs=None,
                cliprange=None,
                loss_normalization=args.loss_normalization,
                normalization_constant=args.normalization_constant,
            )
            train_seconds = time.perf_counter() - train_start

            log_payload: dict[str, Any] = {
                "step": step,
                **scalar_train_metrics(train_metadata),
                "time/weight_sync_seconds": sync_seconds,
                "time/rollout_seconds": rollout_seconds,
                "time/train_seconds": train_seconds,
                "system/max_memory_allocated_gb": torch.cuda.max_memory_allocated(
                    args.train_gpu
                )
                / 2**30,
                "system/max_memory_reserved_gb": torch.cuda.max_memory_reserved(
                    args.train_gpu
                )
                / 2**30,
            }

            should_log_rollouts = (
                args.rollout_log_interval > 0
                and step % args.rollout_log_interval == 0
            )
            if should_log_rollouts:
                train_records = make_generation_records(
                    examples=batch_examples,
                    prompts=batch_prompts,
                    completions=completions,
                    reward_fn=reward_fn,
                    tokenizer=tokenizer,
                    group_size=args.group_size,
                )
                write_jsonl(
                    generations_dir / f"train_step_{step:04d}.jsonl",
                    train_records,
                )
                if wandb_module is not None:
                    log_payload["train/samples"] = wandb_table(
                        wandb_module,
                        train_records,
                        args.wandb_sample_count,
                    )

            should_eval = (
                step % args.eval_interval == 0
                or step == args.num_rollout_steps
            )
            if should_eval:
                eval_sync_start = time.perf_counter()
                server.sync_policy_weights(policy)
                log_payload["time/eval_weight_sync_seconds"] = (
                    time.perf_counter() - eval_sync_start
                )

                eval_start = time.perf_counter()
                val_metrics, val_records = evaluate_policy(
                    server,
                    val_examples,
                    prompt_template,
                    reward_fn,
                    tokenizer,
                    args,
                    use_answer_stop,
                )
                log_payload.update(val_metrics)
                log_payload["time/eval_seconds"] = time.perf_counter() - eval_start
                write_jsonl(
                    generations_dir / f"val_step_{step:04d}.jsonl",
                    val_records,
                )
                if wandb_module is not None:
                    log_payload["val/samples"] = wandb_table(
                        wandb_module,
                        val_records,
                        args.wandb_sample_count,
                    )

            if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
                save_checkpoint(policy, tokenizer, output_dir, f"checkpoint-{step:04d}")

            log_payload["time/step_seconds"] = time.perf_counter() - step_start
            if wandb_run is not None:
                wandb_run.log(log_payload, step=step)
            append_jsonl(
                metrics_path,
                {key: value for key, value in log_payload.items() if isinstance(value, (int, float))},
            )

            message = (
                f"step={step:04d} "
                f"loss={log_payload['train/loss']:.5f} "
                f"train_reward={log_payload['train/reward']:.4f} "
                f"format={log_payload['train/format_reward']:.4f} "
                f"entropy={log_payload['train/token_entropy']:.4f}"
            )
            if "val/reward" in log_payload:
                message += f" val_reward={log_payload['val/reward']:.4f}"
            print(message)

        save_checkpoint(policy, tokenizer, output_dir, "final")
        completed = True
    finally:
        server.stop()
        if wandb_run is not None:
            wandb_run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
