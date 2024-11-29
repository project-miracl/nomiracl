"""
Usage of VLLM generator for efficient batch inference using Ray Data.
Suggested tips for usage:

export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="<your_cache_dir>"
export DATASETS_HF_HOME="<your_cache_dir>"
export HF_HOME="<your_cache_dir>"
"""

from nomiracl.generation import BaseGenerator
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from typing import List, Dict
import ray
import logging
import datasets

logger = logging.getLogger(__name__)


class VLLM(BaseGenerator):
    """VLLM Series Generator for efficient batch inference using Ray Data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Initialize the VLLM generator with model configuration and parameters.
        """
        # Initialize temporary model and delete to avoid memory leak
        temp_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
        )
        del temp_model

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
        )

        # Initialize terminators and stop strings for specific models
        terminators, stop_strings = [], []
        if "llama-3" in self.model_name.lower():
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            stop_strings = ["<|eot_id|>"]

        # Create sampling params
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop_token_ids=terminators if terminators else None,
            stop=stop_strings if stop_strings else None,
        )

    def __call__(self, prompts: List[str], n: int = 1) -> List[str]:
        """
        Generate text from a list of prompts using VLLM.
        """
        # Initialize Ray
        model_name = self.model_name
        max_length = self.max_length
        num_return_sequences = self.num_return_sequences
        cache_dir = self.cache_dir
        sampling_params = self.sampling_params

        # Create a class to do batch inference.
        class LLMPredictor:

            def __init__(self):
                # Create an LLM.
                self.llm = LLM(
                    model=model_name,
                    max_model_len=max_length,
                    max_num_seqs=num_return_sequences,
                    max_seq_len_to_capture=max_length,
                    download_dir=cache_dir,
                    dtype="bfloat16",
                    trust_remote_code=True,
                )  # skip graph capturing for faster cold starts)

            def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
                # Generate texts from the prompts.
                # The output is a list of RequestOutput objects that contain the prompt,
                # generated text, and other information.
                outputs = self.llm.generate(batch["prompt"], sampling_params)
                prompt = []
                generated_text = []
                for output in outputs:
                    prompt.append(output.prompt)
                    generated_text.append(" ".join([o.text for o in output.outputs]))

                return {
                    "prompt": batch["prompt"],
                    "output": generated_text,
                }

        logger.info(f"Initialized VLLM with model {model_name}.")

        # Create Ray Data dataset from prompts
        final_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": f"{prompt}"}]
            prompt_template = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            final_prompts.append({"prompt": prompt_template})

        # convert this list of prompts to a HF dataset
        hf_dataset = datasets.Dataset.from_list(final_prompts)

        # Convert the Huggingface dataset to Ray Data.
        ds = ray.data.from_huggingface(hf_dataset)
        ds = ds.repartition(12, shuffle=False)
        logger.info("Repartitioned dataset into 12 partitions.")

        # Apply batch inference
        ds = ds.map_batches(
            LLMPredictor,
            # Set the concurrency to the number of LLM instances.
            concurrency=self.concurrency,
            # Specify the number of GPUs required per LLM instance.
            # NOTE: Do NOT set `num_gpus` when using vLLM with tensor-parallelism
            # (i.e., `tensor_parallel_size`).
            num_gpus=self.num_gpus,
            # Specify the batch size for inference.
            batch_size=self.batch_size,
            zero_copy_batch=True,
        )

        # Extract results
        results = ds.take_all()
        generated_text = [result["output"] for result in results]
        logger.info(f"Generated text for {len(prompts)} prompts.")
        return generated_text
