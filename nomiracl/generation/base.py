from tqdm.auto import tqdm
from typing import List
import torch


class BaseGenerator:
    def __init__(
        self,
        model_name=None,
        weights_path=None,
        api_key=None,
        cache_dir=None,
        torch_dtype=torch.float16,
        temperature=0.95,
        top_p=0.95,
        num_return_sequences=1,
        max_new_tokens=200,
        min_new_tokens=1,
        max_length=4096,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        device="cuda",
        load_in_8bit=False,
        load_in_4bit=False,
        device_map="auto",
        peft=False,
        concurrency=1,
        num_gpus=1,
        batch_size=1,
        num_partitions=1,
    ):
        self.model_name = model_name
        self.tokenizer = None
        self.weights_path = weights_path
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.device = device
        self.wait = 10
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.peft = peft
        self.concurrency = concurrency
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.num_partitions = num_partitions

    def __call__(self, prompt: str, **kwargs):
        raise NotImplementedError()

    def call(self, prompt: str, **kwargs):
        return self.__call__(prompt, **kwargs)

    def post_process_response(self, response):
        return response

    def batch_call(
        self, prompts: List[str], batch_size: int = 1, **kwargs
    ) -> List[str]:
        batches = [
            prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
        ]

        results = []
        for i, batch in enumerate(
            tqdm(batches, desc="Collecting responses", leave=False)
        ):
            responses = self.__call__(batch, **kwargs)
            results.extend(responses)

        return results

    def truncate_response(self, response: str, max_length: int = 500) -> str:
        if self.tokenizer is None:
            return response
        tokens = self.tokenizer.tokenize(
            response, max_length=max_length, truncation=True
        )
        return self.tokenizer.convert_tokens_to_string(tokens)
