from cs336_alignment.checkpoint import get_model_and_tokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Callable, Literal
from torch.optim import Optimizer

def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer, 
    ) -> dict[str, torch.Tensor]:
    if len(prompt_strs)!=len(output_strs):
        raise ValueError("prompt_strs not equals to output_strs")

    pad_token_id = tokenizer.pad_token_id

    input_ids_list = []
    # labels_list = []
    response_mask_list = []

    prompt_output_lens = []
    # 这个是可以批处理的
    prompt_ids_list = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_ids_list = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list):
        ids = prompt_ids + output_ids

        input_ids_list.append(ids)
        response_mask = ([False]*len(prompt_ids) + [True]*len(output_ids))

        response_mask_list.append(response_mask)
        prompt_output_lens.append(len(ids))

    max_len = max(prompt_output_lens) 
    padded_input_ids = []
    padded_labels = []
    padded_response_masks = []
    # 这里采用先pad 再shift逻辑，是个大坑
    for input_ids, response_mask in zip(input_ids_list, response_mask_list):
        padding_len = max_len - len(input_ids)
        # input_ids = input_ids[1:]
        padding_input_ids = input_ids + [pad_token_id]*padding_len

        padded_input_ids.append(padding_input_ids[:-1])
        padded_labels.append(
            padding_input_ids[1:]
        )
        padded_response_masks.append(response_mask[1:] + [False]*padding_len)

    return {
        "input_ids": torch.tensor(
            padded_input_ids,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            padded_labels,
            dtype=torch.long,
        ),
        "response_mask": torch.tensor(
            padded_response_masks,
            dtype=torch.bool,
        ),
    }

def get_response_log_probs(
    model: PreTrainedModel, input_ids: torch.Tensor, 
    labels: torch.Tensor, return_token_entropy: bool = False, 
    ) -> dict[str, torch.Tensor]:
    """
    需要已经shift的输入
    """
    
    probs = model(input_ids).logits

    log_probs = torch.log_softmax(probs, dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    # 注意交叉熵是对分布做的
    result = {}
    result["log_probs"] = token_log_probs
    # result["token_entropy"] = None
    if(return_token_entropy):
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        result["token_entropy"] = entropy
    return result

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]], 
    rollout_responses: list[str], 
    repeated_ground_truths: list[str], 
    ) -> tuple[torch.Tensor, dict[str, float]]:
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length, "
            f"but got {len(rollout_responses)} and {len(repeated_ground_truths)}."
        )

    if len(rollout_responses) == 0:
        return torch.empty(0, dtype=torch.float32), {
            "mean_reward": 0.0,
            "mean_format_reward": 0.0,
            "mean_answer_reward": 0.0,
        }
    rewards = []
    format_rewards = []
    answer_rewards = []

    required_keys = {"reward", "format_reward", "answer_reward"}
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True):
        reward_result = reward_fn(response, ground_truth)

        missing_keys = required_keys - reward_result.keys()
        if missing_keys:
            raise KeyError(
                f"reward_fn result is missing required keys: {sorted(missing_keys)}"
            )
        # 把这个回答的分数添加到列表
        rewards.append(reward_result["reward"])
        format_rewards.append(reward_result["format_reward"])
        answer_rewards.append(reward_result["answer_reward"])

    raw_rewards = torch.tensor(rewards, dtype = torch.float32)

    metadata = {
        "mean_rewards": raw_rewards.mean().item(),
        "mean_format_reward": torch.tensor(
            format_rewards,
            dtype=torch.float32,
        ).mean().item(),
        "mean_answer_reward": torch.tensor(
            answer_rewards,
            dtype=torch.float32,
        ).mean().item(),
    }

    return raw_rewards, metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    )->tuple[torch.Tensor, dict[str, float]]:
    """
    按组归一化奖励得分
    """

    if raw_rewards.ndim != 1:
        raise ValueError(
            f"raw_rewards must be a 1D tensor, got shape {tuple(raw_rewards.shape)}"
        )
    shape = raw_rewards.shape       # 原则上这里是一维向量
    if(shape[0] % group_size != 0):
        raise ValueError("unvalid group size")
    grouped_raw_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_raw_rewards.mean(dim = -1, keepdim=True)
    group_stds = grouped_raw_rewards.std(dim = -1, keepdim=True)
    if baseline == "mean":
        advantages = (grouped_raw_rewards - group_means) 
    else:
        advantages = grouped_raw_rewards
    if advantage_normalizer == "std":
        advantages /= (group_stds + advantage_eps)
    elif advantage_normalizer == "mean":
        raise NotImplementedError
    else:
        raise NotImplementedError

    advantages = advantages.reshape_as(raw_rewards)
    metadata = {"reward_mean": raw_rewards.mean().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "advantage_mean": advantages.mean().item(),}
    return advantages, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
        如果我理解没错的话，
        raw_rewards_or_advantages: [b,] or [b,1]
        policy_log_probs: [b, max_seq_len]
    """
    if policy_log_probs.ndim != 2:
        raise ValueError(
            "policy_log_probs must have shape "
            f"(batch_size, sequence_length), got {policy_log_probs.shape}"
        )

    if raw_rewards_or_advantages.ndim == 1:
        advantages = raw_rewards_or_advantages.unsqueeze(-1)
    elif (raw_rewards_or_advantages.ndim == 2 and raw_rewards_or_advantages.shape[1] == 1):
        advantages = raw_rewards_or_advantages
    else:
        raise ValueError(
            "raw_rewards_or_advantages must have shape "
            f"(batch_size,) or (batch_size, 1), got "
            f"{raw_rewards_or_advantages.shape}"
        )
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError(
            "Batch size mismatch: "
            f"advantages has batch size {advantages.shape[0]}, "
            f"but policy_log_probs has batch size "
            f"{policy_log_probs.shape[0]}"
        )

    
    metadata: dict[str, torch.Tensor] = {}
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -advantages*policy_log_probs
        return per_token_policy_gradient_loss, metadata
    elif importance_reweighting_method == "grpo":
        raise NotImplementedError
    elif importance_reweighting_method == "gspo":
        raise NotImplementedError
    elif importance_reweighting_method == "noclip":
        raise NotImplementedError

    
def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> torch.Tensor:
    if loss_normalization == "sequence":
        pass
    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(f"normalization: {normalization_constant}")
        raise NotImplementedError

    # 先一个序列内token wise mean 这里bool会在运算时自动转类型
    mask_loss = per_token_policy_gradient_loss * mask
    mask_sum = mask.sum(dim=-1, )
    sequence_loss = mask_loss.sum(dim=-1, ) / mask_sum
    aggregate_loss = sequence_loss.mean(dim=0)

    return aggregate_loss


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    
    if len(repeated_prompts) != len(rollout_responses) or len(repeated_prompts) != len(repeated_ground_truths):
        raise ValueError(
        f"Length mismatch: prompts={len(repeated_prompts)}, "
        f"responses={len(rollout_responses)}, "
        f"ground_truths={len(repeated_ground_truths)}"
    )

    input_len = len(rollout_responses)
    if input_len % group_size != 0:
        raise ValueError("Batch size must be divisible by group_size")
    if gradient_accumulation_steps <= 0 or gradient_accumulation_steps >input_len:
        raise ValueError("unvalid gradient_accumulation_steps ")

    device = next(model.parameters()).device
    # 计算输出的奖励
    raw_rewards, rewards_metadata = compute_rollout_rewards(reward_fn,rollout_responses,repeated_ground_truths)
    # 对奖励进行组归一化
    normed_advantage, advantage_metadata = compute_group_normalized_rewards(raw_rewards,group_size,baseline,advantage_eps,advantage_normalizer)
    normed_advantage = normed_advantage.to(device)
    total_loss = torch.zeros((), device=device)
    
    # 这里要分micro batch 据说是为了防止显存溢出
    microbatch_size = input_len // gradient_accumulation_steps

    entropy_sum = torch.zeros((), device=device)
    response_token_count = torch.zeros((), device=device)
    for i in range(0, input_len, microbatch_size):
        # 记得设成none
        # 选择在循环里分词 防止对全局max len 做pad
        tokenized_result = tokenize_prompt_and_output(repeated_prompts[i:i + microbatch_size],rollout_responses[i:i + microbatch_size],tokenizer)
        # 计算 ids labels 这里记得移动数据 因为分词器一般cpu运行
        input_ids = tokenized_result["input_ids"].to(device)
        labels = tokenized_result["labels"].to(device)
        response_mask  = tokenized_result["response_mask"].to(device)

        # 计算 log probs 这个要model 推理，所以不能放循环外，节省显存
        log_probs, token_entropy =  get_response_log_probs(model,input_ids,labels,return_token_entropy=True).values()
        entropy_sum += (token_entropy * response_mask).sum().detach()
        response_token_count += response_mask.sum().detach()
        # 计算进步加权的梯度，当前仅计算baseline = mean，std，序列归一化的
        if importance_reweighting_method == "none":
            per_token_policy_gradient_loss,loss_metadata =compute_policy_gradient_loss(normed_advantage[i:i + microbatch_size],log_probs,importance_reweighting_method) 
        else:
            raise NotImplementedError
        # 聚合不同序列的损失
        if loss_normalization == "sequence":
            loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss,response_mask,loss_normalization)
        else:
            raise NotImplementedError
        loss = loss*len(input_ids)/input_len
        # 这里要立即反向 以便计算图释放
        total_loss+= loss.detach()
        loss.backward()
    # 梯度剪裁 返回裁剪前的梯度范数
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm is not None else float("inf"),
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mean_token_entropy = entropy_sum / response_token_count.clamp_min(1)

    metadata = {
        **rewards_metadata,
        **advantage_metadata,
        # **loss_metadata,
        "loss": total_loss.item(),
        "grad_norm": grad_norm.detach().item(),
        "token_entropy": mean_token_entropy.detach().item(),
        }
    return total_loss, metadata


    


    
    
    

        


if __name__ == "__main__":
    MODEL_PATH = (
        "/root/.cache/huggingface/hub/"
        "models--allenai--OLMo-2-0425-1B/"
        "snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf"
    )
    prompts = [
        "What is 1+1?", "hello world"
    ]

    outputs = [
        "<think>...</think><answer>2</answer>"
    ]
    model, tokenizer = get_model_and_tokenizer(MODEL_PATH, device="cuda")
    prompts_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

    print(prompts_ids)


    # tokenizer.
    # outputs_ids = tokenizer(outputs, add_special_tokens=False)["input_ids"][0]
    # prompts_and_outputs_ids = prompts_ids+outputs_ids

    # max_prompts_and_outputs_len = max([len(prompts_and_outputs_ids)])
    

    # print(max_prompts_and_outputs_len)