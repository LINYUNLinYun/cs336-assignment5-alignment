# Chunk-level Importance Ratio 实验方案

## 1. 目标与假设

目标：在 GRPO 的 token-level ratio 和 GSPO 的 sequence-level ratio 之间，引入 **chunk-level ratio**，研究 importance weighting 的粒度是否存在更优中间点。

对每个 response，把有效 response token 按连续的 `chunk_size=K` 分组。对第 \(c\) 个 chunk：

$$
s_c=\exp\left(\frac{1}{|C_c|}\sum_{t\in C_c}(\log\pi_\theta(y_t)-\log\pi_0(y_t))\right)
$$

chunk 内所有 token 共用 \(s_c\)，再使用与 GRPO 相同的 clipped surrogate：

$$
L_t=-\min\left(A s_c,\ A\operatorname{clip}(s_c,1-\epsilon,1+\epsilon)\right)
$$

特殊情况：

- `K=1`：退化为 token-level GRPO。
- `K>=response_length`：退化为 sequence-level GSPO 风格的 geometric-mean ratio。

核心假设：**token-level 太局部、方差较大；sequence-level 太全局、会抹平局部差异；中等 chunk size 可能获得更好的稳定性/细粒度信息折中。**

---

## 2. 编程要求

> **重要：不要修改现有 GRPO / GSPO / 训练脚本。必须新开一个 `.py` 文件完成实验，避免影响已有代码和测试。**

建议新建：

```text
cs336_alignment/chunk_ratio_experiment.py
```

要求：

1. 尽量 `import` 现有 tokenizer、reward、rollout、aggregation、logging 等函数；chunk loss 逻辑只放在新脚本中。
2. 新增参数：
   ```bash
   --chunk-size 1/8/32/128/full
   ```
3. 只对 `response_mask == 1` 的 token 分 chunk；最后不足 K 个 token 的 chunk 按实际长度求均值，padding/prompt 不得参与。
4. 除 ratio 粒度外，不改变其他算法设计。

### 必须先做的正确性验证

在正式训练前写简单 tensor test：

- `K=1` 时，loss 与现有 GRPO 在相同 `cliprange` 下数值一致。
- `K>=response_length` 时，loss 与现有 GSPO-style geometric mean 在相同 `cliprange` 下数值一致。
- 检查 variable response length、最后一个残缺 chunk、padding mask。
- `.backward()` 后确认 policy log-prob 有非零梯度。
- 打印一次每个 chunk 的 token 范围、ratio 和 clip 状态，人工检查后关闭 debug。

**上述 sanity check 没通过之前不要开始大规模训练。**

---

## 3. 实验方案

保持现有 off-policy GRPO 实验设置不变：

```text
model: OLMo-2-0425-1B
dataset: GSM8K
prompt: r1_zero
rollout_batch_size: 256
train_batch_size: 8
gradient_accumulation_steps: 1
32x off-policy
```

学习率、生成长度、温度、optimizer、reward、advantage estimator、训练步数等全部沿用当前 baseline。

### Phase A：快速验证

先只跑：

```text
K = 1, 16, full
seed = 42
```

可缩短训练步数，仅检查：

- 是否正常收敛；
- reward 是否明显异常；
- clip fraction 是否接近 0 或 1；
- K=1 的曲线是否和原 GRPO baseline 基本一致。

### Phase B：正式实验

```text
K = 1, 8, 32, 128, full
seed = 42, 43, 44
```

如果算力允许，增加第 4 个 seed。

**主实验必须固定同一个 chunk clipping 规则和同一个 `cliprange`，否则无法判断变化来自 K 还是 clipping。**  
原始 GRPO / GSPO 使用各自默认 cliprange 的结果可以额外作为 reference baseline，但不要和 controlled chunk sweep 混为一谈。

记录：

- validation accuracy / reward（最重要）
- training reward
- clip fraction
- entropy
- response length
- wall-clock time
- 若已有：KL / importance-ratio statistics

---

## 4. 结果分析

至少画：

1. `validation reward vs training step`：不同 K，同一张图，显示多 seed 均值和方差。
2. `final validation reward vs chunk size`。
3. `clip fraction vs training step`。
4. 如果有明显差异，再补 entropy 图。

重点回答：

- 中间 K 是否超过 `K=1` 和 `K=full`？
- K 增大后，seed 间方差是否下降？
- clip fraction 如何随 K 变化？
- 性能提升是否伴随 entropy collapse / response length 异常？
- 最佳 K 是真正稳定提升，还是某个 seed 的偶然结果？

---

## 5. 实验结果（Agent 每次实验后直接更新这里）

> **不要另建结果报告。每次跑完实验后，直接更新本节，并补上 plot 文件路径。**

### 5.1 Sanity Check

- [ ] `K=1 == GRPO` 数值验证通过
- [ ] `K=full == GSPO-style` 数值验证通过
- [ ] mask / variable length 验证通过
- [ ] backward 梯度验证通过

备注：

```text
待填写
```

### 5.2 正式结果

| Chunk size | Seeds | Final Val Reward / Acc (mean±std) | Clip Fraction | Entropy | Wall Time | 备注 |
|---|---|---:|---:|---:|---:|---|
| 1 | 42,43,44 | 待填写 | 待填写 | 待填写 | 待填写 | GRPO endpoint |
| 8 | 42,43,44 | 待填写 | 待填写 | 待填写 | 待填写 | |
| 32 | 42,43,44 | 待填写 | 待填写 | 待填写 | 待填写 | |
| 128 | 42,43,44 | 待填写 | 待填写 | 待填写 | 待填写 | |
| full | 42,43,44 | 待填写 | 待填写 | 待填写 | 待填写 | GSPO-style endpoint |

Plots：

```text
待填写
```

### 5.3 最终结论

Agent 在全部实验结束后填写：

```text
1. 最佳 chunk size：
2. 相对 K=1 的提升：
3. 相对 K=full 的提升：
4. 对 variance / clip fraction / entropy 的影响：
5. 是否支持“中间 importance granularity 更优”的假设：
6. 若结果为负，最可能的原因：
```
