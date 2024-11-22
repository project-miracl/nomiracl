import ray
from vllm import LLM, SamplingParams
from ray.data.dataset import Dataset
from transformers import AutoTokenizer
from typing import List, Dict
import os
import json
import logging

logger = logging.getLogger(__name__)

class VLLM:
    """VLLM Series Generator for efficient batch inference using Ray Data."""

    def __init__(self, model_path: str, cache_dir: str, max_model_len: int, max_new_tokens: int, 
                 temperature: float, concurrency: int, num_gpus: int, batch_size: int):
        """
        Initialize the VLLM generator with model configuration and parameters.
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.num_gpus = num_gpus
        self.batch_size = batch_size

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)

        # Initialize terminators and stop strings for specific models
        self.terminators, self.stop_strings = [], []
        if "llama-3" in model_path.lower():
            self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            self.stop_strings = ["<|eot_id|>"]

        # Create sampling params
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=self.terminators if self.terminators else None,
            stop=self.stop_strings if self.stop_strings else None,
        )

        # Initialize LLM
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            max_num_seqs=1,
            max_seq_len_to_capture=max_model_len,
            download_dir=cache_dir,
            dtype="bfloat16",
            trust_remote_code=True
        )
        logger.info(f"Initialized VLLM with model {model_path}.")

    @staticmethod
    def convert_to_ray_dataset(hf_dataset, num_partitions: int) -> Dataset:
        """
        Convert a HuggingFace dataset to Ray Data and optimize loading.
        """
        logger.info(f"Converting HuggingFace dataset with {len(hf_dataset)} records to Ray Data.")
        
        ds = ray.data.from_huggingface(hf_dataset)
        optimal_partitions = max(num_partitions, ray.available_resources().get("CPU", 1))
        ds = ds.repartition(optimal_partitions, shuffle=False)
        
        logger.info(f"Repartitioned dataset into {optimal_partitions} partitions.")
        return ds

    def apply_batch_inference(self, ds: Dataset) -> Dataset:
        """
        Apply batch inference to a Ray Data dataset using VLLM.
        """
        logger.info(f"Starting batch inference with batch size {self.batch_size} and concurrency {self.concurrency}.")
        
        def predictor(batch: Dict[str, List[str]]) -> Dict[str, list]:
            """
            Predictor function for batch inference.
            """
            outputs = self.llm.generate(batch["prompt"], self.sampling_params)
            generated_text = [
                ' '.join([o.text for o in output.outputs]) for output in outputs
            ]
            return {"prompt": batch["prompt"], "output": generated_text}
        
        ds = ds.map_batches(
            predictor,
            concurrency=min(self.concurrency, ray.available_resources().get("GPU", 1)),
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
            zero_copy_batch=True,
        )
        
        logger.info("Batch inference completed.")
        return ds

    @staticmethod
    def save_results(output_dir: str, results: List[Dict], filename: str = "results.jsonl"):
        """
        Save the results of generated output in JSONL format.
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"Results saved to {os.path.join(output_dir, filename)}.")

    def run(self, hf_dataset, filter_start: int, filter_end: int, output_dir: str, output_filename: str):
        """
        Execute the full pipeline: dataset filtering, conversion, inference, and saving results.
        """
        logger.info(f"Filtering dataset from index {filter_start} to {filter_end}.")
        if filter_end > len(hf_dataset):
            filter_end = len(hf_dataset)
        filtered_dataset = hf_dataset.select(range(filter_start, filter_end))
        
        # Convert to Ray Data
        ray_ds = self.convert_to_ray_dataset(filtered_dataset, num_partitions=12)
        
        # Perform batch inference
        results_ds = self.apply_batch_inference(ray_ds)
        
        # Save results
        results = results_ds.take_all()
        self.save_results(output_dir, results, output_filename)
