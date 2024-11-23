"""
Code to reproduce the baseline evaluation scores in noMIRACL paper (https://arxiv.org/abs/2312.11361).

For gpt-3.5-turbo, gpt-4 (OpenAI) models, add the following lines above your script:
- export OPENAI_API_KEY=your-api-key
- export OPENAI_ORG_ID=your-org-id

For gpt-3.5-azure, gpt-4-azure models, add the following lines above your script:
Your API would look something of this form: <AZURE_OPENAI_API_BASE>openai/deployments/<AZURE_DEPLOYMENT_NAME>/chat/completions?api-version=<AZURE_OPENAI_API_VERSION>
- export AZURE_OPENAI_API_VERSION = "your-azure-openai-api-version"
- export AZURE_OPENAI_API_BASE = "your-azure-opnai-api-base"
- export AZURE_DEPLOYMENT_NAME = "your-azure-deployment-name"

For all other open-sourced models available in HuggingFace, you can directly use the code below.

Usage example:

for lang in ar bn de en es fa fr fi hi id ja ko ru sw te th yo zh; do
for model in gemma-7b-it
do
    CUDA_VISIBLE_DEVICES=0,1 python repro_nomiracl_generation.py --language $lang --split test --subsets non_relevant \
    --prompt_template vanilla \
    --model_name $model \
    --output_dir ../results/baselines-new \
    --filename $lang.test.yesno_prompt.regex \
    --relevant_ratio 0.01 --non_relevant_ratio 0.99 \
    --batch_size 1 --temperature 0.1 --top_p 0.95 --max_new_tokens 50 --max_passage_tokens 370

    CUDA_VISIBLE_DEVICES=0,1 python repro_nomiracl_generation.py --language $lang --split test --subsets relevant \
    --prompt_template vanilla \
    --model_name $model \
    --output_dir ../results/baselines-new \
    --filename $lang.test.yesno_prompt.regex \
    --relevant_ratio 0.99 --non_relevant_ratio 0.01 \
    --batch_size 1 --temperature 0.1 --top_p 0.95 --max_new_tokens 50 --max_passage_tokens 370
done done

"""

from nomiracl.dataset import NoMIRACLDataLoader
from nomiracl import util, LoggingHandler
from nomiracl.generation.utils import load_model
from nomiracl.prompts.utils import load_prompt_template
from tqdm.autonotebook import tqdm

import argparse
import os
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
    parser.add_argument("--language", default=None)
    parser.add_argument("--prompt_template", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--results_dir", default="../results/baselines")
    parser.add_argument("--filename", default=False)
    parser.add_argument("--overwrite", default=False, action="store_true")
    # dataset specific arguments
    parser.add_argument("--data_dir", default=None, required=False)
    parser.add_argument("--split", default="test")
    parser.add_argument("--subsets", default="non_relevant", nargs="+")
    parser.add_argument("--relevant_ratio", type=float, default=0.5)
    parser.add_argument("--non_relevant_ratio", type=float, default=0.5)
    parser.add_argument("--max_sample_pool", required=False, type=int, default=None)
    # model specific arguments
    parser.add_argument("--model_name", default="flan-t5-xxl")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", required=False, type=float, default=0.3)
    parser.add_argument("--top_p", required=False, type=float, default=0.95)
    parser.add_argument("--max_new_tokens", required=False, type=int, default=200)
    parser.add_argument("--max_passage_tokens", required=False, type=int, default=380)
    parser.add_argument("--max_length", required=False, type=int, default=4096)

    args = parser.parse_args()

    ##############
    # Load Model #
    ##############

    model_name = args.model_name
    logging.info("Loading model: {}...".format(model_name))

    # Load model from HuggingFace, provide the complete name under the weights path of the model
    models_supported = [
        "vicuna",
        "llama",
        "olmo",
        "mistral",
        "mixtral",
        "orca",
        "phi",
        "zephyr",
        "bloom",
        "flan",
        "aya",
        "gemma",
    ]

    if any(model_type in model_name.lower() for model_type in models_supported):
        if "llama" in model_name:
            weights_path = f"meta-llama/{model_name.capitalize()}-hf"
        elif "vicuna" in model_name:
            weights_path = f"lmsys/{model_name}"
        elif "mixtral" in model_name.lower() or "mistral" in model_name.lower():
            weights_path = f"mistralai/{model_name}"
        elif "orca" in model_name.lower() or "phi" in model_name.lower():
            weights_path = f"microsoft/{model_name}"
        elif "zephyr" in model_name.lower():
            weights_path = f"HuggingFaceH4/{model_name}"
        elif "bloom" in model_name.lower():
            weights_path = f"bigscience/{model_name}"
        elif "flan" in model_name.lower():
            weights_path = f"google/{model_name}"
        elif "aya" in model_name.lower():
            weights_path = f"CohereforAI/{model_name}"
        elif "gemma" in model_name.lower():
            weights_path = f"google/{model_name}"

    # You can also provide loading in 8bit and 4bit format
    load_in_8bit, load_in_4bit = False, False

    # Add all model parameters when loading the model
    # 1. weights path contains the huggingface model name
    # 2. cache_dir is the directory to save the model cache
    # 3. max_new_tokens is the maximum number of tokens to generate
    # 4. max_length is the maximum length of the input prompt
    # 5. temperature is the temperature for sampling
    # 6. top_p is the top-p for sampling
    # 7. device_map is the device to load the model (auto is best for multiple GPU loading)
    # 8. load_in_8bit is the flag to load the model in 8bit format (bitsandbytes)
    # 9. load_in_4bit is the flag to load the model in 4bit format (bitsandbytes)
    model = load_model(
        model_name,
        weights_path=weights_path,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )

    # Add model parameters
    logging.info("Loaded model: {}...".format(model_name))
    logging.info("Model temperature: {}...".format(model.temperature))
    logging.info("Model top_p: {}...".format(model.top_p))
    logging.info("Model max_new_tokens: {}...".format(model.max_new_tokens))
    logging.info("Model max length: {}...".format(model.max_length))

    #########################
    # Load NoMIRACL dataset #
    #########################
    language_code_map = {
        "ar": "arabic",
        "bn": "bengali",
        "de": "german",
        "en": "english",
        "es": "spanish",
        "fa": "persian",
        "fi": "finnish",
        "fr": "french",
        "hi": "hindi",
        "id": "indonesian",
        "ja": "japanese",
        "ko": "korean",
        "ru": "russian",
        "sw": "swahili",
        "te": "telugu",
        "th": "thai",
        "yo": "yoruba",
        "zh": "chinese",
    }

    data_loader = NoMIRACLDataLoader(
        language=language_code_map[args.language],
        split=args.split,
        hf_dataset_name="miracl/nomiracl",
        load_from_huggingface=True,
    )

    corpus, queries, qrels = data_loader.load_data_sample(
        relevant_ratio=args.relevant_ratio,
        non_relevant_ratio=args.non_relevant_ratio,
        max_sample_pool=args.max_sample_pool,
    )

    ###########################
    # 3. Generate Prompt Data #
    ###########################

    # Loading Prompt Template
    max_passage_count = 10
    prompt_cls = load_prompt_template(args.prompt_template, count=max_passage_count)

    for subset in args.subsets:
        # Final preprocessed prompts
        input_prompts, results = {}, {}

        ### This code is added for reproduction, to generate on the same queries are generated used in paper () ####
        # Use the same queries as generated in the paper
        input_filepath = os.path.join(
            args.results_dir,
            subset,
            f"{args.language}.{args.split}.{args.prompt_template}_prompt.jsonl",
        )
        results = util.load_results_as_jsonl(input_filepath=input_filepath)
        MODEL_NAME = "gpt-4-azure"
        generate_query_ids = list(results[MODEL_NAME].keys())

        # check how many queries we need to generate additionally
        if not args.overwrite:
            if model_name in results:
                updated_list = []
                for query_id in generate_query_ids:
                    if query_id not in results[model_name]:
                        updated_list.append(query_id)

                generate_query_ids = updated_list

        # check whether already queries are generated for the subset
        if len(generate_query_ids) == 0:
            logging.info(f"Queries already generated for {subset} for {model_name}...")
            continue
        else:
            logging.info(
                f"Loaded {list(results.keys())} models from subset: {subset} in {input_filepath}..."
            )

            # Generate prompts

            final_query_ids = []
            separator = ": "

            for query_id in tqdm(
                generate_query_ids,
                total=len(generate_query_ids),
                desc=f"Processing {args.language} queries",
            ):

                if (
                    query_id in queries[subset]
                    and len(qrels[subset][query_id]) == max_passage_count
                ):
                    doc_ids = [
                        doc_id for doc_id in qrels[subset][query_id] if doc_id in corpus
                    ]

                    if len(doc_ids) == max_passage_count:
                        passage_list = []
                        query = queries[subset][query_id]
                        for doc_id in qrels[subset][query_id]:
                            if doc_id in corpus:
                                passage = f"{corpus.get(doc_id).get('title')}{separator}{corpus[doc_id].get('text')}"
                                passage_list.append(passage)
                            else:
                                logging.error(f"Doc {doc_id} not found in corpus...")

                    passages_truncated = [
                        model.truncate_response(p, max_length=args.max_passage_tokens)
                        for p in passage_list
                    ]
                    prompt = prompt_cls(query=query, passages=passages_truncated)
                    input_prompts[query_id] = prompt
                    final_query_ids.append(query_id)

            logging.info(
                f"Generating responses for {len(input_prompts)} {subset} queries..."
            )
            if model_name not in results:
                results[model_name] = {}

            for query_id_chunk in tqdm(
                util.chunks(final_query_ids, args.batch_size),
                total=len(final_query_ids) // args.batch_size,
                desc=f"Processing {args.language} with {model_name}",
            ):

                prompt_list = [input_prompts[query_id] for query_id in query_id_chunk]
                model_results = model.batch_call(
                    prompt_list, batch_size=args.batch_size
                )

                # Save model result for each query in model_result
                for idx, query_id in enumerate(query_id_chunk):
                    model_result = model_results[idx]
                    results[model_name][query_id] = model_result

                # # Save results
                output_dir = os.path.join(args.output_dir, subset)
                os.makedirs(output_dir, exist_ok=True)

                util.save_results_as_jsonl(
                    output_dir=output_dir,
                    results=results,
                    qrels=qrels[subset],
                    prompts=input_prompts,
                    template=args.prompt_template,
                    filename=f"{args.language}.{args.split}.{args.prompt_template}_prompt.jsonl",
                )
