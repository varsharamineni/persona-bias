import argparse
from datetime import datetime
import json
import os
import time

from persona.dataset import get_dataset
from persona.evaluators import get_evaluator
from persona.prompts import get_prompt
from persona.models import get_model, get_api_key, get_org_id

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

### CHANGE: vLLM imports
from vllm import LLM, SamplingParams


def extract_text_from_output(output_text):
    return output_text.strip()

### vLLM loading function
def load_vllm_model(model_name_or_path, model_params, tensor_parallel_size):
    print(f"Loading vLLM model {model_name_or_path}...")
    llm = LLM(model=model_name_or_path, dtype="float16", tensor_parallel_size=tensor_parallel_size)  
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


### vLLM wrapper class
class VLLMModelWrapper:
    def __init__(self, llm, tokenizer, model_params):
        self.llm = llm
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.usage = {"tokens_used": 0}

    def generate(self, user_prompt, system_prompt, stop_condition=None, idx=None):
    # Build chat messages instead of concatenating strings
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply Qwen chat template with thinking enabled
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=model_params.get("enable_thinking", True))

        sampling_params = SamplingParams(
            temperature=model_params.get("temperature", 0.6),
            top_p=model_params.get("top_p", 0.95),
            top_k=model_params.get("top_k", 20),
            min_p=model_params.get("min_p", 0.0),
            max_tokens=model_params.get("max_output_tokens", 512),
            stop=model_params.get("stop_sequences", None),
            presence_penalty=model_params.get("presence_penalty", 0.0),
            stop_token_ids=model_params.get("eos_token_id", None),
        )

        # outputs = []
        # for output in self.llm.chat(messages, sampling_params=sampling_params):
        #     outputs.append(output[0].content)
        # text = outputs[0]
        # return text

        outputs = self.llm.chat([messages], sampling_params=sampling_params, 
        chat_template_kwargs={"enable_thinking": True}, use_tqdm=False)
        output = outputs[0]
        text = output.outputs[0].text
        return text


        # outputs = self.llm.generate([prompt], sampling_params)
        # text = outputs[0].outputs[0].text
        # return text

    def add_usage(self, raw_response):
        num_tokens = len(self.tokenizer.encode(raw_response))
        self.usage["tokens_used"] += num_tokens

    def extract_text(self, raw_response):
        return extract_text_from_output(raw_response)

    def get_name(self):
        return model_name

    def print_usage(self, num_examples):
        print(f"Total tokens used: {self.usage['tokens_used']} for {num_examples} examples")
        print(f"Average tokens per example: {self.usage['tokens_used'] / max(1, num_examples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##Dataset
    parser.add_argument("--dataset_name", default='mmlu-college_biology')
    parser.add_argument("--dataset_path", default='')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    ##Model config
    parser.add_argument("--api_key", default="")
    parser.add_argument("--org_id", default="")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613")
    parser.add_argument("--model_config_path", default="")

    # Model config
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                    help="Number of tensor parallel partitions for vLLM")

    ##Prompt config
    parser.add_argument("--prompt_type", default="adopt_identity_accordance")
    parser.add_argument("--persona", default="a Human")

    ##Output config
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--cached_path", default="")
    parser.add_argument("--experiment_prefix", default="")
    parser.add_argument("--output_dir", default="results")

    ##Evaluation config
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--evaluator_name", default="")



    args = parser.parse_args()

    print("Arguments:")
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}", end="\n")

    dataset_name = args.dataset_name
    dataset_path = None

    if ("mmlu" not in dataset_name) and ("bbh" not in dataset_name) \
            and ("arc" not in dataset_name) and (dataset_name != 'mbpp'):
        if args.dataset_path:
            dataset_path = args.dataset_path
        else:
            if dataset_name != "default":
                dataset_path = f"data/{dataset_name}/test.jsonl"
            else:
                raise ValueError("Unable to figure out dataset path")
        if not os.path.isfile(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        else:
            print(f"Loading dataset from {dataset_path}...")

    if ("arc" in dataset_name) and ("_" in dataset_name):
        grades = [grade.strip() for grade in dataset_name.split("_")[1:]]
        partition = dataset_name.split("_", 1)[0]
        dataset = get_dataset(partition, dataset_path, keep_grades=set(grades))
    else:
        dataset = get_dataset(dataset_name, dataset_path)

    start_idx = args.start_idx
    end_idx = args.end_idx
    test_set = dataset.get_data()
    if start_idx > 0 or end_idx > 0:
        if (end_idx < 0) or (end_idx > len(test_set)):
            end_idx = len(test_set)
        test_set = test_set[start_idx:end_idx]

    print(f"Using {len(test_set)} instances from dataset ({start_idx} to {end_idx})")
    print(f"First instance: {test_set[0]}")

    model_config_path = args.model_config_path
    model_name = args.model_name

    if not model_config_path:
        print(f"No model config path provided. Using default config for {model_name}")
        model_config_path = f"configs/{model_name}/default.json"

    if os.path.isfile(model_config_path):
        print(f"Loading model config from {model_config_path}")
        with open(model_config_path, "r") as fr:
            model_params = json.load(fr)
        if model_params["model_name"] != model_name:
            print(f"WARNING: model name in config ({model_params['model_name']}) != provided ({model_name})")
            model_params["model_name"] = model_name
    else:
        raise ValueError(f"Model config path {model_config_path} does not exist.")

    ### use vLLM else use api 
    llm, tokenizer = load_vllm_model(model_name, model_params, args.tensor_parallel_size)
    model = VLLMModelWrapper(llm, tokenizer, model_params)
    print(f"Done initializing vLLM model: {model_name}")
 
    print("Model params:")
    for param_name in model_params:
        print(f"{param_name}: {model_params[param_name]}")

    prompt_type = args.prompt_type
    persona = args.persona
    prompt_dict = get_prompt(dataset_name, model_name, prompt_type, persona)

    now = datetime.now()
    dt_string = now.strftime("%m-%d_%Hh-%Mm_%Ss_%fms")

    if "mmlu" in dataset_name:
        task_name = dataset_name.split("-")[1].strip()
        output_dir = f"results/mmlu/{model_name}/{prompt_type}/{task_name}"
    elif "bbh" in dataset_name:
        task_name = dataset_name.split("-")[1].strip()
        output_dir = f"results/bbh/{model_name}/{prompt_type}/{task_name}"
    elif "arc" in dataset_name:
        partition = dataset_name
        grades = "all"
        if "_" in dataset_name:
            partition = dataset_name.split("_", 1)[0].strip()
            grades = dataset_name.split("_", 1)[1].strip()
        output_dir = f"results/{partition}/{model_name}/{prompt_type}/{grades}"
    else:
        output_dir = f"results/{dataset_name}/{model_name}/{prompt_type}"

    os.makedirs(output_dir, exist_ok=True)

    if args.experiment_prefix:
        persona_fname = persona.replace(" ", "_")
        raw_output_path = f"{output_dir}/{args.experiment_prefix}_{persona_fname}_raw_responses.jsonl"
        text_output_path = f"{output_dir}/{args.experiment_prefix}_{persona_fname}_text_predictions.jsonl"
    else:
        persona_fname = persona.replace(" ", "_")
        raw_output_path = f"{output_dir}/{persona_fname}_raw_responses.jsonl"
        text_output_path = f"{output_dir}/{persona_fname}_text_predictions.jsonl"

    cached_set = dict()
    if args.cached_path:
        with open(args.cached_path, 'r') as f:
            for line in f:
                cached_set[json.loads(line)['id']] = line
        print(f"Loaded {len(cached_set)} cached predictions")
    else:
        print("No cached predictions provided.")

    num_cache_hits = 0
    num_api_errors = 0
    t0 = time.time()
    with open(raw_output_path, 'w', encoding='utf-8') as raw_writer, \
         open(text_output_path, 'w', encoding='utf-8') as text_writer:
        for i, instance in enumerate(test_set):
            question_id = instance["id"]
            if question_id in cached_set:
                num_cache_hits += 1
                text_writer.write(cached_set[question_id].strip("\n") + "\n")
                continue

            question = instance["question"]
            if dataset_name == "mbpp":
                system_prompt = prompt_dict["system_prompt"]
                user_prompt = prompt_dict["user_prompt_builder"](user_prompt=prompt_dict["user_prompt"], question=question, tests=instance["tests"])
            else:
                system_prompt = prompt_dict["system_prompt"]
                user_prompt = prompt_dict["user_prompt"].format(question=question)

            if args.dry_run:
                print(f"System prompt: {system_prompt}")
                print(f"User prompt: {user_prompt}")
                print('=======================')
                continue

            raw_response = model.generate(user_prompt=user_prompt, system_prompt=system_prompt,
                                          stop_condition=dataset.stop_condition, idx=question_id)
            if raw_response is None:
                raw_response = ""
                num_api_errors += 1

            model.add_usage(raw_response)
            raw_out_data = {
                "id": question_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw_response
            }
            raw_writer.write(json.dumps(raw_out_data) + "\n")

            text_responses = model.extract_text(raw_response)
            instance['predicted_explanations'] = text_responses
            text_writer.write(json.dumps(instance) + "\n")

            if (i + 1) % 10 == 0:
                print(f"Finished {i + 1} examples", flush=True)

    print(f"Finished {len(test_set)} examples")
    print(f"Cache hits: {num_cache_hits}")
    print(f"API errors: {num_api_errors}")

    model.print_usage(len(test_set))
    seconds_taken = time.time() - t0
    print(f"Total time: {seconds_taken:.2f}s")
    print(f"Avg per example: {seconds_taken / len(test_set):.2f}s")

    if args.eval:
        print("Evaluating predictions ...")
        evaluator_name = args.evaluator_name if args.evaluator_name else dataset_name
        evaluator = get_evaluator(evaluator_name)
        evaluator.evaluate(text_output_path)
