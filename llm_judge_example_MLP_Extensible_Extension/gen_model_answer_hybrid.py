"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template

# Medusa imports
import transformers


from medusa.model.medusa_model_hybrid import MedusaModel
from medusa.model.medusa_choices import *

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    top_p,
    sampling,
    fast,
    medusa_choices,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                posterior_threshold,
                posterior_alpha,
                sampling,
                top_p,
                fast,
                medusa_choices,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    sampling,
    top_p,
    fast,
    medusa_choices,
):
    
    # Medusa model setup
    
    num_heads = -1
    for choice in medusa_choices:
        if len(choice) > num_heads:
            num_heads = len(choice)

    model = MedusaModel.from_pretrained(
        model_path,
        # medusa_num_heads = num_heads,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:',model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    
    question = questions[0]

    # warmup
    for _ in range(3):
        conv = get_conversation_template(model_id)
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            device = next(model.parameters()).device
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            #input_ids = tokenizer([prompt]).input_ids #, return_tensors="pt").input_ids.to(model.base_model.device)

            try:
                torch.cuda.synchronize()
                start_time = time.time()
                
                # Use medusa_generate (proven stable method)
                output_text = ""
                for step in model.medusa_generate(
                    input_ids,
                    temperature=0.7,
                    max_steps=max_new_token,
                    medusa_choices=medusa_choices,
                    posterior_threshold=posterior_threshold,
                    posterior_alpha=posterior_alpha,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                ):
                    output_text = step["text"]
                
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                
                output = output_text

                # Count tokens in the generated output
                num_generated_tokens = len(tokenizer.encode(output, add_special_tokens=False))
                
                # Apply conversation-specific post-processing
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print(e)
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            conv.messages[-1][-1] = output
    print('Warmup done')


    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            conv = get_conversation_template(model_id)
            turns = []
            wall_time = []
            num_tokens = []
            num_generated_tokens = 0
            
            num_decoding_array = []
            for j in range(len(question["turns"])):
                num_decoding_steps = 0
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                device = next(model.parameters()).device
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                #input_ids = tokenizer([prompt]).input_ids #, return_tensors="pt").input_ids.to(model.base_model.device)

                # some models may error out when generating long outputs
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    # Use medusa_generate (proven stable method)
                    output_text = ""
                    for step in model.medusa_generate(
                        input_ids,
                        temperature=temperature,
                        max_steps=max_new_token,
                        medusa_choices=medusa_choices,
                        posterior_threshold=posterior_threshold,
                        posterior_alpha=posterior_alpha,
                        top_p=top_p,
                        sampling=sampling,
                        fast=fast,
                    ):
                        output_text = step["text"]
                        num_decoding_steps += 1
                    
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    
                    output = output_text

                    # Count tokens in the generated output
                    num_generated_tokens = len(tokenizer.encode(output, add_special_tokens=False))

                    # Apply conversation-specific post-processing
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"
                    total_time = 0.0

                turns.append(output)
                wall_time.append(total_time)
                num_tokens.append(num_generated_tokens)
                num_decoding_array.append(num_decoding_steps)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "wall_time": wall_time, "num_tokens": num_tokens, "num_decoding_steps": num_decoding_array})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    # YL: Medusa args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )

    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for medusa sampling.",
    )
    
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for medusa sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="The top-p for medusa sampling.",
    )

    parser.add_argument(
        "--sampling",
        type=str,
        default="typical",
        help="The sampling method for medusa sampling.",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Whether to use fast decoding.",
    )

    parser.add_argument(
        "--medusa-choices",
        type=str,
        default="mc_sim_7b_63",
        help="The medusa choices for medusa sampling.",
    )

    


    args = parser.parse_args()

    args.model_id = args.model_id+"-temperature-"+str(args.temperature)+"-posterior_threshold-"+str(args.posterior_threshold)+"-posterior_alpha-"+str(args.posterior_alpha)+"-top_p-"+str(args.top_p)+"-sampling-"+args.sampling+"-fast-"+str(args.fast)
    args.medusa_choices = eval(args.medusa_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,

        args.temperature,
        args.posterior_threshold,
        args.posterior_alpha,
        args.top_p,
        args.sampling,
        args.fast,
        args.medusa_choices,
    )

    reorg_answer_file(answer_file)

