import argparse
import subprocess
import time
from persona.dataset.mmlu import MMLU

predef_datasets = []
predef_datasets.extend(["college_mathematics", "abstract_algebra"])  # MMLU - Maths datasets
predef_datasets.extend(["college_physics", "conceptual_physics"])  # MMLU - Physics datasets
predef_datasets.extend(["college_chemistry", "high_school_chemistry"])  # MMLU - Chemistry datasets
predef_datasets.extend(["college_computer_science", "computer_security", "machine_learning"])  # MMLU - CS datasets
predef_datasets.extend(["moral_scenarios", "human_sexuality", "professional_medicine"])
predef_datasets.extend(["professional_law", "professional_accounting", "professional_psychology"])
predef_datasets.extend(["high_school_government_and_politics", "high_school_world_history"])
predef_datasets.extend(["college_biology", "anatomy"])
predef_datasets.extend(["sociology", "logical_fallacies", "management"])
predef_datasets = ["mmlu-" + dataset for dataset in predef_datasets]
predef_datasets.extend(["mbpp"])
predef_datasets.extend(["bbh-sports_understanding"])

predef_persona_map = {
    "shared": [
        "a physically-disabled person", "an able-bodied person",
        "an Atheist person", "a Religious person", "a Jewish person", "a Christian person",
        "an Asian person", "a Hispanic person", "an African person", "a Caucasian person",
        "a lifelong Democrat", "a lifelong Republican", "a Donald Trump supporter", "a Barack Obama supporter",
        "a man", "a woman",
        "an average human", "a human",
    ],
}

predef_max_size_map = {
    "mmlu-moral_scenarios": 250,
    "mmlu-professional_medicine": 250,
    "gsm8k": 250,
    "mmlu-professional_law": 250,
    "mmlu-professional_accounting": 250,
    "mmlu-professional_psychology": 250
}

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument("--org_id", default="")
parser.add_argument("--datasets")
parser.add_argument("--personas")
parser.add_argument("--use_predef_personas", action="store_true", default=False)
parser.add_argument("--use_predef_datasets", action="store_true", default=False)
parser.add_argument("--use_predef_max_size", action="store_true", default=False)
parser.add_argument("--run_no_persona", action="store_true", default=False)
parser.add_argument("--prompt_type", default="no_persona")
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
parser.add_argument("--experiment_prefix", default="")
parser.add_argument("--out_file_prefix", default="")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--model_name", default="gpt-3.5-turbo-0613")
parser.add_argument("--model_path", default="", help="Path to local HF model for vLLM")
parser.add_argument("--use_local_model", action="store_true", help="Whether to use local vLLM model")

if __name__ == "__main__":
    args = parser.parse_args()

    print("Arguments:")
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")

    # Determine datasets
    datasets = predef_datasets if args.use_predef_datasets else [d.strip() for d in args.datasets.split(',')]
    print(f"Using datasets: {datasets}")

    # Build dataset-persona map
    dataset_persona_map = {}
    if args.use_predef_personas:
        for dataset in datasets:
            dataset_persona_map[dataset] = []
            dataset_persona_map[dataset].extend(predef_persona_map.get("shared", []))
            dataset_persona_map[dataset].extend(predef_persona_map.get(dataset, []))
            if args.run_no_persona:
                dataset_persona_map[dataset].append("no_persona")
    else:
        if (args.personas is None) and (not args.run_no_persona):
            raise ValueError("No persona has been specified")
        personas = [p.strip() for p in args.personas.split(',')] if args.personas else []
        for dataset in datasets:
            dataset_persona_map[dataset] = personas.copy()
            if args.run_no_persona:
                dataset_persona_map[dataset].append("no_persona")

    print("Dataset-Persona map:")
    for dataset, personas in dataset_persona_map.items():
        print(f"{dataset}: {personas}")

    # Parallelization parameters
    parallelization_factor = 2
    sleep_time = 15 * 60  # 15 minutes

    # Launch scripts
    for dataset, personas in dataset_persona_map.items():
        print(f"\n\nLaunching scripts for dataset: {dataset}")
        done_personas = 0
        for persona in personas:
            done_personas += 1
            print(f"\nLaunching for dataset: {dataset}, persona: {persona}")

            prompt_type = args.prompt_type if persona != "no_persona" else "no_persona"

            kwargs = {
                "dataset_name": dataset,
                "prompt_type": prompt_type,
                "persona": f'"{persona}"',
                "model_name": args.model_name
            }
            if args.org_id:
                kwargs['org_id'] = args.org_id
            if args.start_idx is not None:
                kwargs['start_idx'] = args.start_idx
            if args.use_predef_max_size:
                kwargs['end_idx'] = predef_max_size_map.get(dataset, -1)
                print(f"Using max size {kwargs['end_idx']} for dataset {dataset}")
            elif args.end_idx is not None:
                kwargs['end_idx'] = args.end_idx
            if args.experiment_prefix:
                kwargs['experiment_prefix'] = args.experiment_prefix

            # Build output file name
            base_output = f"{prompt_type}_{dataset}_{persona.replace(' ', '_')}_{args.model_name}"
            if 'end_idx' in kwargs and kwargs['end_idx'] != -1:
                base_output = f"{prompt_type}_{dataset}_size{kwargs['end_idx']}_{persona.replace(' ', '_')}_{args.model_name}"
            output_file = f"{args.out_file_prefix}_{base_output}.txt" if args.out_file_prefix else f"{base_output}.txt"

            # Build command
            args_string = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
            args_string += " --eval"
            if args.use_local_model:
                args_string += " --use_local_model"
            if args.model_path:
                args_string += f" --model_path {args.model_path}"

            for i in range(args.repeat):
                out_file = output_file if args.repeat == 1 else output_file.replace(".txt", f"_r{i}.txt")
                command = f"nohup python -u persona/run_vllm.py {args_string} > {out_file} 2>&1 &"
                print(f"Run {i+1}, command: {command}")
                subprocess.Popen(command, shell=True)
                time.sleep(3)

            if done_personas % parallelization_factor == 0:
                print(f"Done {done_personas} personas, sleeping {sleep_time / 60:.1f} mins")
                time.sleep(sleep_time)

        print(f"Finished dataset {dataset}, sleeping {sleep_time / 60:.1f} mins")
        time.sleep(sleep_time)
