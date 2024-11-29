"""
Usage Example with HF: 
CUDA_VISIBLE_DEVICES=0 python sample_model_generation.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python sample_model_generation.py

For usage with VLLM:
export RAY_DEDUP_LOGS=0 # Disable deduplication of logs
export RAY_TMPDIR="<your_cache_dir>" # Directory to save the Ray logs (else /tmp/ray/)
export DATASETS_HF_HOME="<your_cache_dir>"
export HF_HOME="<your_cache_dir>"
export CUDA_VISIBLE_DEVICES=0,1 python sample_model_generation.py
"""

from nomiracl.generation.utils import load_model
from nomiracl import LoggingHandler

import logging

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Load model from HuggingFace, provide the complete name under the weights path of the model
model_name = "llama-3-8B-Instruct"  # Model name to load
weights_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Weights path for the model
temperature = 0.3  # Temperature for sampling
top_p = 0.95  # Top-p for sampling
max_new_tokens = 200  # Maximum number of new tokens to generate
cache_dir = "/mnt/users/n3thakur/cache"  # Directory to save the model cache

logging.info("Loading model: {}...".format(model_name))

model = load_model(
    "huggingface",  # run_type
    model_name,  # model_name
    weights_path=weights_path,  # extra kwargs
    cache_dir=cache_dir,  # extra kwargs
    temperature=temperature,  # extra kwargs
    top_p=top_p,  # extra kwargs
    max_new_tokens=max_new_tokens,  # extra kwargs
)

# for vllm, we provide the following extra kwargs:
# batch_size, num_gpus, concurrency.
# model = load_model(
#     "vllm", # run_type
#     weights_path, # model_name
#     cache_dir="<your-cache-dir>", # extra kwargs
#     batch_size=2, # extra kwargs - set based on max length
#     num_gpus=1, # extra kwargs - set as 1 always
#     concurrency=2 # extra kwargs - set as count(GPUs)
# )

logging.info("Loaded model: {}...".format(model_name))
logging.info("Model temperature: {}...".format(model.temperature))
logging.info("Model top_p: {}...".format(model.top_p))
logging.info("Model max_new_tokens: {}...".format(model.max_new_tokens))

# Sample prompts
prompts = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
]

model_results = model.batch_call(prompts, batch_size=1)
# for vllm, since we already provide batch-size use call instead of batch_call
# model_results = model.call(prompts)

for prompt, result in zip(prompts, model_results):
    logging.info("Prompt: {}".format(prompt))
    logging.info("{} result: {}".format(model_name, result))
