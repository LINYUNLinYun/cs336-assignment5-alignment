## 3 Prompting
总结了下不同提示词的base model的表现：

| Prompt             |     Correct |    Category2 |    Category3 |
| ------------------ | ----------: | -----------: | -----------: |
| question_only      |    3 (0.2%) |  215 (16.3%) | 1101 (83.5%) |
| r1_zero            |           0 |  797 (60.4%) |  522 (39.6%) |
| r1_zero_three_shot | 218 (16.5%) | 1042 (79.0%) |    59 (4.5%) |


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


