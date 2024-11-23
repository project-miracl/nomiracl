"""
Usage example: python sample_vanilla_prompt.py
"""

from nomiracl.dataset import NoMIRACLDataLoader
from nomiracl import util, LoggingHandler
from nomiracl.prompts.utils import load_prompt_template
from tqdm.autonotebook import tqdm

import logging, os

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

language = "en"  # Language of the dataset
split = "test"  # Split of the dataset: dev, test
relevant_ratio = 0.5  # Ratio of relevant samples
non_relevant_ratio = 0.5  # Ratio of non-relevant samples
max_sample_pool = 250  # Maximum cap of samples to load for each subset


# Technique 1: Load the NoMIRACL dataset from Huggingface
# NoMIRACL (HuggingFace): https://huggingface.co/datasets/miracl/nomiracl
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
    language=language_code_map[language],
    split=split,
    hf_dataset_name="miracl/nomiracl",
    load_from_huggingface=True,
)

corpus, queries, qrels = data_loader.load_data_sample(
    relevant_ratio=relevant_ratio,
    non_relevant_ratio=non_relevant_ratio,
    max_sample_pool=max_sample_pool,
)


## Load the vanilla prompt template
prompt_template_name = "vanilla"
subset = "non_relevant"  # Subset of the dataset: relevant, non_relevant
separator = ": "  # Separator between title and text in the passage
max_count = 10  # Maximum number of passages

prompt_cls = load_prompt_template(
    prompt_template_name, count=max_count
)  # count denotes the maximum number of passages
query_ids_list = list(queries[subset].keys())
prompts = []

for query_id in tqdm(
    query_ids_list, total=len(query_ids_list), desc=f"Processing {language} queries"
):
    if query_id in queries[subset] and len(qrels[subset][query_id]) == max_count:
        doc_ids = [doc_id for doc_id in qrels[subset][query_id] if doc_id in corpus]

        # If there are 10 documents for the query, then create a prompt
        if len(doc_ids) == max_count:
            passage_list = []
            query = queries[subset][query_id]
            for doc_id in qrels[subset][query_id]:
                if doc_id in corpus:
                    passage = f"{corpus.get(doc_id).get('title')}{separator}{corpus[doc_id].get('text')}"
                    passage_list.append(passage)
                else:
                    logging.info(f"Doc {doc_id} not found in corpus...")

        prompt = prompt_cls(query=query, passages=passage_list)
        prompts.append(prompt)

for prompt in prompts[:2]:
    logging.info(f"Prompt: {prompt}")
