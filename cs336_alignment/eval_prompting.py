import json
from collections import Counter
from pathlib import Path

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from pathlib import Path

from cs336_alignment.drgrpo_grader import extract_answer


def load_gsm8k(path):
    data = []

    with open(path) as f:
        for line in f:
            item = json.loads(line)

            question = item["question"]

            # 只取 #### 后面的答案
            answer = item["answer"].split("####")[-1].strip()

            data.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )

    return data

def load_prompt(path):
    with open(path) as f:
        return f.read()

def build_prompts(template:str, questions):
    return [
        template.format(question=q)
        for q in questions
    ]

def main():
    data = load_gsm8k('data/gsm8k/test.jsonl')
    # print(data[:10])
    # return 
    # 启动vllm服务
    server = VLLMServer(
        model_id="/root/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf",
        gpu=0,)

    server.start()



    prompts = {
        "question_only":
            ("cs336_alignment/prompts/question_only.prompt",
            question_only_reward_fn),
        "r1_zero":
            ("cs336_alignment/prompts/r1_zero.prompt",
            r1_zero_reward_fn),
        "r1_zero_three_shot":
            ("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
            r1_zero_reward_fn),
    }

    for name, (prompt_path, reward_fn) in prompts.items():
        oringinal_prompt = load_prompt(prompt_path)

        prompt_list = build_prompts(oringinal_prompt, [x['question'] for x in data])

        sampling_params = {
             "n":1, 
             "seed":42, 
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512,
        }
        if 'r1_zero' in name:
            # 走r1的评测
            sampling_params["stop"] = ["</answer>"]
            sampling_params["include_stop_str_in_output"] = True
        outputs = server.generate_completions(prompt_list, sampling_params)
        responses = [x.text for x in outputs]
        results = evaluate(responses, data, reward_fn)

        print("="*50)
        print(name)
        print(results)




def evaluate(responses, examples, reward_fn, save_path : Path = 'cs336_alignment/results'):
    save_path = Path(save_path)
    if save_path is None:
        return 
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    category_2_f = open(save_path / 'category_2_answers.json', 'w')
    category_3_f = open(save_path / 'category_3_answers.json', 'w')

    counter = Counter()

    for response, example in zip(responses, examples):

        result = reward_fn(
            response,example["answer"]
        )

        model_answer = response
        if "</think> <answer>" in response and "</answer>" in response:
            model_answer = response.split("<answer>")[-1].replace("</answer>", "")
            if "\\boxed" in model_answer:
                model_answer = extract_answer(model_answer)
        item = {
            # "question": example["question"],
            "ground_truth": example["answer"],
            "response": model_answer,
            # "format_reward": result["format_reward"],
            # "answer_reward": result["answer_reward"],
            # "reward": result["reward"],
        }
        if (
            result["format_reward"] == 1
            and result["answer_reward"] == 1
        ):
            counter["format_correct_answer_correct"] += 1

        elif (
            result["format_reward"] == 1
            and result["answer_reward"] == 0
        ):
            counter["format_correct_answer_wrong"] += 1
            category_2_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        else:
            counter["format_wrong_answer_wrong"] += 1
            category_3_f.write(json.dumps(item, ensure_ascii=False) + "\n")
    category_2_f.close()
    category_3_f.close()
    return counter

if __name__ == '__main__':
    main()



