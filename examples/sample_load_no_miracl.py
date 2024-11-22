"""
Usage example: python load_no_miracl.py
"""

from nomiracl.dataset import NoMIRACLDataLoader
from nomiracl import util, LoggingHandler

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

# Technique 1: Download the NoMIRACL dataset directly from the web and unzip

data_dir = "/mnt/users/n3thakur/research_new/miracl-unanswerable/no-miracl"  # Directory to save the dataset

data_dir = os.path.join(data_dir, language)
data_loader = NoMIRACLDataLoader(data_dir=data_dir, split=split)

corpus, queries, qrels = data_loader.load_data_sample(
    relevant_ratio=relevant_ratio,
    non_relevant_ratio=non_relevant_ratio,
    max_sample_pool=max_sample_pool,
)


# Technique 2: Load the NoMIRACL dataset from Huggingface
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
