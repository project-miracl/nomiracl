# Initial code structure taken from https://github.com/McGill-NLP/instruct-qa.

from math import inf
from typing import List
import time, os
import tiktoken
import openai
from openai import (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    AuthenticationError,
    Timeout,
)

import torch
import requests
import logging
from transformers import pipeline
from tqdm.autonotebook import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BloomForCausalLM,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


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

    def __call__(self, prompt, **kwargs):
        raise NotImplementedError()

    def post_process_response(self, response):
        return response

    def batch_call(self, prompts, batch_size=1, **kwargs):
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

    def truncate_response(self, response, max_length=500):
        if self.tokenizer is None:
            return response
        tokens = self.tokenizer.tokenize(
            response, max_length=max_length, truncation=True
        )
        return self.tokenizer.convert_tokens_to_string(tokens)


#################################
# Azure GPT-3.5/GPT-4 Generator #
#################################
class GPTxAzure(BaseGenerator):
    """OpenAI GPT-3.5-turbo and GPT-4 models on Azure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # API: <AZURE_OPENAI_API_BASE>openai/deployments/<AZURE_DEPLOYMENT_NAME>/<model_map[model_name]>/completions?api-version=<AZURE_OPENAI_API_VERSION>
        # where model_map = {"gpt-3.5-turbo": "chat", "gpt-4": "chat"}
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
        AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

        self.model_name = self.model_name.replace("-azure", "")
        self.model_map = {
            "gpt-3.5-turbo": "chat",
            "gpt-4": "chat",
        }

        assert (
            self.model_name in self.model_map
        ), "You should add the model name to the model -> endpoint compatibility mappings."
        assert self.model_map[self.model_name] in [
            "chat",
            "completions",
        ], "Only chat and completions endpoints are implemented. You may want to add other configurations."

        # json error happens if max_new_tokens is inf
        map_name = self.model_map[self.model_name]
        self.api_url = f"{AZURE_OPENAI_API_BASE}openai/deployments/{AZURE_DEPLOYMENT_NAME}/{map_name}/completions?api-version={AZURE_OPENAI_API_VERSION}"
        self.max_new_tokens = self.max_new_tokens

    def __call__(self, prompts, n=1):
        responses = []
        for prompt in prompts:
            headers = {"Content-Type": "application/json", "api-key": self.api_key}

            response = self.api_request(
                prompt=prompt,
                headers=headers,
                n=n,
            )
            if n == 1:
                responses.append(response[0])
            else:
                responses.append(response)
        return responses

    def api_request(self, prompt, headers, n):
        try:
            if self.model_map[self.model_name] == "chat":
                res = requests.post(
                    url=self.api_url,
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "n": n,
                    },
                )
                if "choices" not in res.json():

                    if "code" in res.json()["error"]:
                        if res.json()["error"]["code"] == "429":
                            logger.error(
                                f"Error: {res.json()}. Waiting {self.wait} seconds before retrying."
                            )
                            time.sleep(self.wait)
                            return self.api_request(prompt, headers, n)
                        else:
                            logger.error(f"Error: {res.json()}. Do not retry!")
                            return ["Unable to generate response."]
                else:
                    return [r["message"]["content"] for r in res.json()["choices"]]
        except (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            logger.error(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self.api_request(prompt, headers, n)

    def truncate_response(self, response, max_length=500):
        encoder = tiktoken.encoding_for_model(self.model_name)
        input_tokens = encoder.encode(response)
        if len(input_tokens) <= max_length:
            return response
        else:
            return encoder.decode(input_tokens[:max_length])


##################################
# OpenAI GPT-3.5/GPT-4 Generator #
##################################
class GPTx(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        openai.api_key = OPENAI_API_KEY
        try:
            OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
            openai.organization = OPENAI_ORGANIZATION
        except:
            pass

        self.model_map = {
            "gpt-3.5-turbo": "chat",
            "gpt-4": "chat",
        }
        assert (
            self.model_name in self.model_map
        ), "You should add the model name to the model -> endpoint compatibility mappings."
        assert self.model_map[self.model_name] in [
            "chat",
            "completions",
        ], "Only chat and completions endpoints are implemented. You may want to add other configurations."
        # json error happens if max_new_tokens is inf
        self.max_new_tokens = self.max_new_tokens

    def __call__(self, prompts, n=1):
        responses = []
        for prompt in prompts:
            kwargs = {"temperature": self.temperature, "top_p": self.top_p, "n": n}
            if self.max_new_tokens != inf:
                kwargs["max_tokens"] = self.max_new_tokens
            response = self.api_request(
                prompt,
                **kwargs,
            )
            if n == 1:
                responses.append(response[0])
            else:
                responses.append(response)
        return responses

    def api_request(self, prompt, **kwargs):
        try:
            if self.model_map[self.model_name] == "chat":
                res = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                return [r.message.content for r in res.choices]
            elif self.model_map[self.model_name] == "completions":
                res = openai.Completion.create(
                    model=self.model_name, prompt=prompt, **kwargs
                )
                return [r.text for r in res.choices]
        except (
            RateLimitError,
            APIConnectionError,
            InternalServerError,
            NotFoundError,
            PermissionDeniedError,
            AuthenticationError,
            Timeout,
        ) as e:
            logger.error(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self.api_request(prompt, **kwargs)

    def truncate_response(self, response: str, max_length: int = 500) -> str:
        encoder = tiktoken.encoding_for_model(self.model_name)
        input_tokens = encoder.encode(response)
        if len(input_tokens) <= max_length:
            return response
        else:
            return encoder.decode(input_tokens[:max_length])


######################################
# HuggingFace LLAMA Series Generator #
######################################
class Llama(BaseGenerator):
    """Llama-2, Vicuna, Alpaca, Orca AutoModelForCausalLM models can be loaded using Llama class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, padding_side="left"
        )

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            bnb_8bit_quant_type="nf4",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
            bnb_8bit_compute_dtype=self.torch_dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        if not self.load_in_8bit and not self.load_in_4bit:
            self.model = self.model.bfloat16()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts: List[str]) -> List[str]:

        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        generate_ids = self.model.generate(
            input_ids=_input.input_ids,
            attention_mask=_input.attention_mask,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )

        return [
            self.tokenizer.decode(
                generate_ids[i, _input.input_ids.size(1) :],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            for i in range(generate_ids.size(0))
        ]


###############################
# HuggingFace BLOOM Generator #
###############################
class BLOOM(BaseGenerator):
    """BLOOM (BloomForCausalLM) model from BigScience."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, cache_dir=self.cache_dir
        )

        self.model = BloomForCausalLM.from_pretrained(
            self.weights_path,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts: List[str]) -> List[str]:

        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        generate_ids = self.model.generate(
            input_ids=_input.input_ids,
            attention_mask=_input.attention_mask,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        return [
            self.tokenizer.decode(
                generate_ids[i, _input.input_ids.size(1) :],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            for i in range(generate_ids.size(0))
        ]


###################################
# HuggingFace Flan, AYA Generator #
###################################
class Flan(BaseGenerator):
    """Flan, Aya AutoModelForSeq2SeqLM models can be loaded using Flan class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, cache_dir=self.cache_dir
        )

        # Error happens if max_new_tokens is inf
        self.tokenizer.model_max_length = self.max_length

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.weights_path,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )

    def __call__(self, prompts: List[str]) -> List[str]:

        _input = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.device
        )

        generate_ids = self.model.generate(
            _input.input_ids,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        return self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
        )


##################################
# HuggingFace Pipeline Generator #
##################################
class Pipeline(BaseGenerator):
    """Pipeline model from HuggingFace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline = pipeline(
            model=f"{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=self.device_map,
        )

        self.pipeline.tokenizer.pad_token_id = 0

    def forward_call(self, prompt: str) -> str:
        _input = self.pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )

        return self.pipeline(
            self.pipeline.tokenizer.decode(
                _input.input_ids[0], attention_mask=_input.attention_mask[0]
            ),
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )[0]["generated_text"]

    def __call__(self, prompts: List[str]) -> List[str]:
        return [self.forward_call(prompt) for prompt in prompts]


#############################
# HuggingFace Falcon Models #
#############################
class Falcon(BaseGenerator):
    """Falcon model from TI2 (Running via pipeline)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.torch_dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.weights_path}")

        self.pipeline = pipeline(
            "text-generation",
            model=f"{self.weights_path}",
            tokenizer=self.tokenizer,
            model_kwargs={"cache_dir": self.cache_dir},
            torch_dtype=self.torch_dtype,
            load_in_8bit=self.load_in_8bit,
            trust_remote_code=True,
            device_map=self.device_map,
        )

    def forward_call(self, prompt: str) -> str:
        sequences = self.pipeline(
            prompt,
            max_length=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]["generated_text"]

    def __call__(self, prompts: List[str]) -> List[str]:
        return [self.forward_call(prompt) for prompt in prompts]


######################################
# HuggingFace Mistral/Mixtral Models #
######################################
class Mistral(BaseGenerator):
    """Mistral/Mixtral model from MistralAI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_dtype = torch.float16

        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, trust_remote_code=True
        )
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            bnb_8bit_quant_type="nf4",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
            bnb_8bit_compute_dtype=self.torch_dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts: List[str]) -> List[str]:
        final_prompts = []

        # Apply chat template to the prompts
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            final_prompts.append(prompt)

        _input = self.tokenizer(
            final_prompts,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)

        response = self.model.generate(
            input_ids=_input.input_ids,
            attention_mask=_input.attention_mask,
            do_sample=True,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        return [
            self.tokenizer.decode(
                response[i, _input.input_ids.size(1) :],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            for i in range(response.size(0))
        ]

    def truncate_response(self, response: str, max_length: int = 500) -> str:
        tokens = self.tokenizer.tokenize(response)[:max_length]
        return self.tokenizer.convert_tokens_to_string(tokens)


##########################################
# HuggingFace AutoModelForCausalLM Model #
##########################################
class HFAutoModelCausalLM(BaseGenerator):
    """Model class for generic HuggingFace AutoModelForCausalLM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_dtype = torch.float16

        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, trust_remote_code=True
        )

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            bnb_8bit_quant_type="nf4",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
            bnb_8bit_compute_dtype=self.torch_dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts: List[str]) -> List[str]:
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)

        response = self.model.generate(
            input_ids=_input.input_ids,
            attention_mask=_input.attention_mask,
            do_sample=True,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        return [
            self.tokenizer.decode(
                response[i, _input.input_ids.size(1) :],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            for i in range(response.size(0))
        ]

    def truncate_response(self, response: str, max_length: int = 500) -> str:
        tokens = self.tokenizer.tokenize(response)[:max_length]
        return self.tokenizer.convert_tokens_to_string(tokens)
