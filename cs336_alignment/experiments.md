## 3 Prompting
总结了下不同提示词的base model的表现：
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


