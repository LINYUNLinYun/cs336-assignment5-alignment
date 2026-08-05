## 3 Prompting
总结了下不同提示词的base model的表现：

| Prompt             |     Correct |    Category2 |    Category3 |
| ------------------ | ----------: | -----------: | -----------: |
| question_only      |    3 (0.2%) |  215 (16.3%) | 1101 (83.5%) |
| r1_zero            |           0 |  797 (60.4%) |  522 (39.6%) |
| r1_zero_three_shot | 218 (16.5%) | 1042 (79.0%) |    59 (4.5%) |
类别三我就不多看了，毕竟连格式都错了，不过也有少部分是答案对只是格式错导致答案没被分割出来的。

主要看看类别2的：
1. zero-r1 
   1. 答案区域包含 reasoning （最多），在答案里继续推理
   2. 数学错误
   3. 思路对了，但是格式错
   4. 幻觉问题
2. zero-r1-three-shots
   1. 数学计算错误
   2. 理解错误
   3. 格式错
   4. 幻觉，提前结束等
3. question-only
   1. 模型没有遵守 boxed answer 格式
   2. 模型复述题目 / 继续生成
   3. box 提取失败
所以，按照正确率来看，three-shots > question-only > zero-r1（question-only对了三个，zero-r1一个没对）

few-shot 的提示词效果最好，对了两百多个，即使是错的，也多是格式或模型能力问题。

尤其是加入了 few-shot 的提示词，模型的回答中很少reasoning，说明 few-shot 的提示词可以引导模型进行更好的回答。

而question-only 的提示词，虽然对了三个，但是类别三有一千多个，说明它的格式遵循能力是很差的。zero-r1 的提示词，虽然没有对的，但是类别二比较多，说明它的格式遵循能力还行。但总的说few-shot 的提示词效果最好。

类别三的格式错误：
- </think> \n<answer> 多了一个或多个换行符导致识别失败的
- 说一通结果没有\boxed{...}
- </think>缺失，或生成错误
- 生成了 </think> <answer>，但没有闭合 </answer>。
- 幻觉，触发最大ctx等

总之，仅通过提示词的还是比较局限的，下面看强化学习的。


## 4 Group Relative Policy Optimization

### 4.1 Driving on-policy GRPO
#### problem(a): Policy gradient estimator variance

定义单个样本：

$$
Z=r(A)\nabla_\theta \log \pi_\theta(A)
$$

则：

$$
\hat g=\frac{1}{n}\sum_{i=1}^{n}Z_i
$$

由于 $Z_i$ 独立同分布：

$$
\mathrm{Var}(\hat g)=\frac{\mathrm{Var}(Z)}{n}
$$

因此只需要计算 $\mathrm{Var}(Z)$。


策略：

$$
\pi_\theta(A=1)=p=\sigma(\theta)
$$

因此：

$$
\pi_\theta(A=0)=1-p
$$

奖励函数：

$$
r(A)=\mathbb{1}\{A=1\}
$$

当 $A=1$ 时：
$$
r(A)=1
$$

且：

$$
\nabla_\theta\log\pi_\theta(A=1)
=
\nabla_\theta\log p
$$

因为：

$$
\frac{dp}{d\theta}=p(1-p)
$$

所以：

$$
\nabla_\theta\log p
=
\frac{1}{p}\frac{dp}{d\theta}
=
\frac{1}{p}p(1-p)
=
1-p
$$

因此：

$$
Z_1=1\cdot(1-p)=1-p
$$

发生概率：

$$
P(Z=1-p)=p
$$

当 $A=0$时：
$$
r(A)=0
$$

所以：

$$
Z_0=0
$$

发生概率：

$$
P(Z=0)=1-p
$$



因此随机变量 $Z$：

| $Z$ | Probability |
|---|---|
| $1-p$ | $p$ |
| $0$ | $1-p$ |

期望：

$$
E[Z]
=
p(1-p)+(1-p)0
$$

$$
=p(1-p)
$$

二阶矩与方差：

$$
E[Z^2]
=
p(1-p)^2+(1-p)0^2
$$

$$
=p(1-p)^2
$$


$$
\mathrm{Var}(Z)
=
E[Z^2]-E[Z]^2
$$

代入：

$$
=
p(1-p)^2-[p(1-p)]^2
$$
化简：

$$
\mathrm{Var}(Z)=p(1-p)^3
$$

最终：

$$
\boxed{
\mathrm{Var}(\hat g)
=
\frac{p(1-p)^3}{n}
}
$$

#### Problem(b): Variance reduction with baseline

定义单个样本：

$$
Z=(r(A)-b)\nabla_\theta\log\pi_\theta(A)
$$

当 $A=1$：

$$
\nabla_\theta\log\pi_\theta(A=1)=1-p
$$

因此：

$$
Z_1=(1-b)(1-p)
$$

当 $A=0$：

$$
\nabla_\theta\log\pi_\theta(A=0)=-p
$$

因此：

$$
Z_0=(-b)(-p)=bp
$$

因此随机变量 $Z$：

| $Z$ | Probability |
|---|---|
| $(1-b)(1-p)$ | $p$ |
| $bp$ | $1-p$ |


计算期望：

$$
E[Z]
=
p(1-b)(1-p)+(1-p)bp
$$

$$
=p(1-p)((1-b)+b)
$$

$$
=p(1-p)
$$


二阶矩：

$$
E[Z^2]
=
p(1-b)^2(1-p)^2+(1-p)b^2p^2
$$


方差：

$$
\mathrm{Var}(Z)=E[Z^2]-(E[Z])^2
$$

$$
=
p(1-b)^2(1-p)^2+(1-p)b^2p^2-p^2(1-p)^2
$$

化简：

$$
=
p(1-p)(b-(1-p))^2
$$

因此：

$$
\boxed{
\mathrm{Var}(\hat g)
=
\frac{p(1-p)(b-(1-p))^2}{n}
}
$$

#### Problem(c): When baseline = p
$$
\mathrm{Var}(\hat g) = \frac{p(1-p)(2p-1)^2}{n}
$$

和无baseline的方差比较：当p大于2/3时，baseline的方差更大；当p小于2/3时，baseline的方差更小。所以baseline的选择需要根据p的大小来决定，不是一直都能起到降低方差的作用。

### 4.2 Implementing on-policy GRPO

