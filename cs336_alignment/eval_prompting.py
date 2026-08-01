import json
import shutil
from collections import Counter
from pathlib import Path

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn,
    extract_answer,
)


MODEL_PATH = (
    "/root/.cache/huggingface/hub/"
    "models--allenai--OLMo-2-0425-1B/"
    "snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf"
)


def load_gsm8k(path):
    data = []

    with open(path) as f:
        for line in f:
            item = json.loads(line)

            data.append({
                "question": item["question"],
                "answer": item["answer"].split("####")[-1].strip(),
            })

    return data

def load_prompt(path):
    with open(path) as f:
        return f.read()

def build_prompts(template, questions):
    return [
        template.format(question=q)
        for q in questions
    ]

def evaluate(
    responses,
    examples,
    reward_fn,
    result_dir,
):
    result_dir = Path(result_dir)

    if result_dir.exists():
        shutil.rmtree(result_dir)

    result_dir.mkdir(parents=True)

    category2_file = result_dir / "category2.jsonl"
    category3_file = result_dir / "category3.jsonl"

    counter = Counter()

    with open(category2_file, "w") as f2, open(category3_file, "w") as f3:

        for response, example in zip(responses, examples):

            result = reward_fn(
                response,
                example["answer"],
            )

            if (
                result["format_reward"] == 1
                and result["answer_reward"] == 1
            ):
                counter["correct"] += 1
                continue


            if (
                result["format_reward"] == 1
                and result["answer_reward"] == 0
            ):
                counter["category2"] += 1

                model_answer = response

                if "<answer>" in response and "</answer>" in response:
                    model_answer = (
                        response.split("<answer>")[-1].replace("</answer>", "").strip()
                    )

                    if "\\boxed" in model_answer:
                        model_answer = extract_answer(model_answer)


                f2.write(
                    json.dumps(
                        {
                            # "question": example["question"],
                            "model_answer": model_answer,
                            "ground_truth": example["answer"],
                            # "full_response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


            else:
                counter["category3"] += 1

                f3.write(
                    json.dumps(
                        {
                            # "question": example["question"],
                            "ground_truth": example["answer"],
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


    with open(result_dir / "summary.json", "w") as f:
        json.dump(counter, f, indent=2)


    return counter



def main():

    data = load_gsm8k("data/gsm8k/test.jsonl")

    # 注意：
    # 这里不会 start()
    # 因为 vLLM 已经独立启动
    server = VLLMServer(
        model_id=MODEL_PATH,
    )


    prompts = {
        "question_only": (
            "cs336_alignment/prompts/question_only.prompt",
            question_only_reward_fn,
            False,
        ),

        "r1_zero": (
            "cs336_alignment/prompts/r1_zero.prompt",
            r1_zero_reward_fn,
            True,
        ),

        "r1_zero_three_shot": (
            "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
            r1_zero_reward_fn,
            True,
        ),
    }


    for name, (prompt_path, reward_fn, is_r1) in prompts.items():

        template = load_prompt(prompt_path)

        prompt_list = build_prompts(
            template,
            [
                x["question"]
                for x in data
            ],
        )


        sampling_params = {
            "n": 1,
            "seed": 42,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512,
        }


        if is_r1:
            sampling_params["stop"] = [
                "</answer>"
            ]

            sampling_params[
                "include_stop_str_in_output"
            ] = True


        outputs = server.generate_completions(
            prompt_list,
            sampling_params,
        )


        responses = [
            x.text
            for x in outputs
        ]


        result = evaluate(
            responses,
            data,
            reward_fn,
            f"cs336_alignment/results/eval_prompts/{name}",
        )


        print("=" * 50)
        print(name)
        print(result)



if __name__ == "__main__":
    main()