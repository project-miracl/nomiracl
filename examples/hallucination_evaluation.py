"""
Usage example:
python hallucination_evaluation.py --languages ar bn de en es fa fr fi hi id ja ko ru sw te th yo zh \
--split test --subsets relevant non_relevant \
--prompt_template vanilla \
--output_dir ../results/eval \
--results_dir ../results/baselines \
--output_filename test.vanilla_prompt.all \
--parse_models gpt-4-azure gpt-3.5-azure Mixtral-8x7B-Instruct-v0.1 Mistral-7B-Instruct-v0.2 Orca-2-13b Orca-2-7b aya-101 llama-2-70b-chat llama-2-13b-chat llama-2-7b-chat flan-t5-xxl
"""

from nomiracl import util, LoggingHandler
from nomiracl.eval.metrics import ConfusionMetric
from nomiracl.prompts.utils import load_prompt_template

import argparse
import os
import csv
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default=None, nargs="+")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--results_dir", default="../results/baselines")
    parser.add_argument("--prompt_template", default="vanilla")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subsets", default="relevant", nargs="+")
    parser.add_argument("--output_filename", default=None)
    parser.add_argument("--parse_models", default=None, nargs="+")
    args = parser.parse_args()

    # Evaluate the results for each subset: relevant, non_relevant
    for subset in args.subsets:

        # Loading Prompt Template
        prompt_cls = load_prompt_template(args.prompt_template, count=10)
        total_models = len(args.parse_models)
        error_string = "Error" if subset == "relevant" else "Hallucination"

        os.makedirs(args.output_dir, exist_ok=True)
        output_filepath = os.path.join(
            args.output_dir, f"eval_results.{subset}.{args.output_filename}.tsv"
        )

        with open(output_filepath, "w") as f:
            writer = csv.writer(f, delimiter="\t")

            # Write the header of the CSV file
            writer.writerow(
                ["Language"]
                + ["Accuracy"] * total_models
                + [error_string] * total_models
                + ["Invalid"] * total_models
            )
            writer.writerow(["Code"] + args.parse_models * 3)

            # Evaluate the results for each language in NoMIRACL
            for language in args.languages:
                eval_results, postproc_results = {}, {}

                # Load if previously stored results and prompts
                input_filepath = os.path.join(
                    args.results_dir,
                    subset,
                    f"{language}.{args.split}.{args.prompt_template}_prompt.jsonl",
                )
                if os.path.exists(input_filepath):
                    logging.info(f"Loaded results from {input_filepath}...")
                    results, postproc_results = util.load_results_as_jsonl(
                        input_filepath=input_filepath
                    )
                else:
                    ValueError(f"No results found at {input_filepath}...")

                ####################################
                # 1. Postprocess each model output #
                ####################################
                for model_key, output in results.items():
                    if model_key in args.parse_models:
                        query_ids = list(output.keys())
                        postprocess_results = [
                            prompt_cls.postprocess(model_output)
                            for model_output in output.values()
                        ]
                        postproc_results[model_key] = {
                            query_id: p_result
                            for query_id, p_result in zip(
                                query_ids, postprocess_results
                            )
                        }

                ##################################
                # 2. Compute hallucination stats #
                ##################################
                if args.prompt_template == "vanilla":
                    evaluator = ConfusionMetric()
                    eval_results = evaluator.evaluate(subset, postproc_results)

                model_results = []

                for metric in ["Accuracy", error_string, "Invalid"]:
                    for model_name in args.parse_models:
                        if model_name in eval_results[metric]:
                            model_results.append(
                                round(eval_results[metric][model_name] * 100, 4)
                            )
                        else:
                            model_results.append(0.0)

                writer.writerow([language] + model_results)
