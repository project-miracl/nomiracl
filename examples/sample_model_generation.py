"""
Usage Example: CUDA_VISIBLE_DEVICES=0 python sample_model_generation.py
Multiple GPUs Usage Example: CUDA_VISIBLE_DEVICES=0,1,2,3 python sample_model_generation.py
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

model_name = "zephyr-7b-beta"  # Model name to load

temperature = 0.3  # Temperature for sampling
top_p = 0.95  # Top-p for sampling
max_new_tokens = 200  # Maximum number of new tokens to generate
cache_dir = "/mnt/users/n3thakur/cache"  # Directory to save the model cache


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

    model = load_model(model_name, weights_path=weights_path, cache_dir=cache_dir)

# Add model parameters
model.temperature = temperature
model.top_p = top_p
model.max_new_tokens = max_new_tokens

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

for prompt, result in zip(prompts, model_results):
    logging.info("Prompt: {}".format(prompt))
    logging.info("{} result: {}".format(model_name, result))
