# Code idea and initial development from https://github.com/McGill-NLP/instruct-qa.

from math import inf
from typing import List
import time, os
import tiktoken
import openai
from openai.error import (
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
    APIError,
    Timeout,
)

import torch
import requests
from transformers import pipeline
from tqdm.autonotebook import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BloomForCausalLM,
    OPTForCausalLM,
)


MAX_LENGTH = 4096


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
        max_new_tokens=200,
        min_new_tokens=1,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        device="cuda",
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

    def __call__(self, prompt, **kwargs):
        raise NotImplementedError()

    def post_process_response(self, response):
        return response
    
    def batch_call(self, prompts, batch_size=1, **kwargs):
        batches = [
            prompts[i : i + batch_size]
            for i in range(0, len(prompts), batch_size)
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
        tokens = self.tokenizer.tokenize(response, max_length=max_length, truncation=True)
        return self.tokenizer.convert_tokens_to_string(tokens)

class GPTxAzure(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
        self.api_key = os.getenv("OPEN_AI_API_KEY")
        self.model_name = self.model_name.replace("-azure", "")
        self.model_map = {
            "gpt-3.5-turbo": "chat",
            "gpt-4": "chat",
        }
        self.deployment_map = {
            "gpt-3.5-turbo": "gpt35t",
            "gpt-4": "gpt4all",
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
        deployment_name = self.deployment_map[self.model_name]
        self.api_url = f"{AZURE_OPENAI_API_BASE}openai/deployments/{deployment_name}/{map_name}/completions?api-version={AZURE_OPENAI_API_VERSION}"
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
                    json={"messages": [{"role": "user", "content": prompt}], 
                          "max_tokens": self.max_new_tokens, 
                          "temperature": self.temperature, 
                          "top_p": self.top_p, 
                          "n": n
                        },
                )
                if "choices" not in res.json():
                    
                    if "code" in res.json()["error"]:
                        if res.json()["error"]["code"] == '429':
                            print(f"Error: {res.json()}. Waiting {self.wait} seconds before retrying.")
                            time.sleep(self.wait)
                            return self.api_request(prompt, headers, n)
                        else:
                            print(f"Error: {res.json()}. Do not retry")
                            return ["Unable to generate response."]
                else:
                    return [r["message"]["content"] for r in res.json()["choices"]]
        except (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self.api_request(prompt, headers, n)
    
    def truncate_response(self, response, max_length=500):
        encoder = tiktoken.encoding_for_model(self.model_name)
        input_tokens = encoder.encode(response)
        if len(input_tokens) <= max_length:
            return response
        else:
            return encoder.decode(input_tokens[:max_length])


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
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:
            print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self.api_request(prompt, **kwargs)
    
    def truncate_response(self, response, max_length=500):
        encoder = tiktoken.encoding_for_model(self.model_name)
        input_tokens = encoder.encode(response)
        if len(input_tokens) <= max_length:
            return response
        else:
            return encoder.decode(input_tokens[:max_length])
    
    def get_tokens(self, response):
        encoder = tiktoken.encoding_for_model(self.model_name)
        return encoder.encode(response)

class Llama(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, padding_side="left"
        )
        if "70b" in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.weights_path,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                load_in_8bit=True,
                cache_dir=self.cache_dir
            )
            # self.model = self.model.bfloat16()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.weights_path, 
                torch_dtype=self.torch_dtype, 
                device_map="auto",
                cache_dir=self.cache_dir
            )
            self.model = self.model.bfloat16()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
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

    def post_process_response(self, response):
        keywords = {"user:", "User:", "assistant:", "- Title:", "Question:"}
        end_keywords = {"Agent:", "Answer:"}

        response_lines = response.split("\n")
        response_lines = [x for x in response_lines if x.strip() not in ["", " "]]

        for j, line in enumerate(response_lines):
            if any(line.startswith(kw) for kw in keywords):
                response_lines = response_lines[:j]
                break

        for j, line in enumerate(response_lines):
            if j > 0 and any(line.startswith(kw) for kw in end_keywords):
                response_lines = response_lines[:j]
                break

        return "\n".join(response_lines)

class BLOOM(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, cache_dir=self.cache_dir)
        self.model = BloomForCausalLM.from_pretrained(
            self.weights_path,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=MAX_LENGTH,
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

class Vicuna(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"lmsys/{self.model_name}", cache_dir=self.cache_dir, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            f"lmsys/{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=MAX_LENGTH,
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


class OPT(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"facebook/{self.model_name}", cache_dir=self.cache_dir, padding_side="left"
        )
        self.model = OPTForCausalLM.from_pretrained(
            f"facebook/{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=MAX_LENGTH,
            truncation=True,
        ).to(self.device)
        generate_ids = self.model.generate(
            **_input,
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


class Flan(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "aya" in self.model_name: 
            model_name = f"CohereForAI/{self.model_name}"
        else:
           model_name = f"google/{self.model_name}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
        )
        
        self.tokenizer.model_max_length = MAX_LENGTH
        
        if "ul2" in self.model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            load_in_8bit=True,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                device_map="auto",
            )

    def __call__(self, prompts):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

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


class PipelineGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline(
            model=f"{self.model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = 0

    def forward_call(self, prompt):
        _input = self.pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=MAX_LENGTH,
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

    def __call__(self, prompts):
        return [self.forward_call(prompt) for prompt in prompts]

class FalconPipelineGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.torch_dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(f"tiiuae/{self.model_name}")
        
        if "40b" in self.model_name:
            self.pipeline = pipeline(
                "text-generation",
                model=f"tiiuae/{self.model_name}",
                tokenizer=self.tokenizer,
                model_kwargs={"cache_dir": self.cache_dir},
                torch_dtype=self.torch_dtype,
                load_in_8bit=True,
                trust_remote_code=True,
                device_map="auto",
        )
        else:
            self.pipeline = pipeline(
                "text-generation",
                model=f"tiiuae/{self.model_name}",
                tokenizer=self.tokenizer,
                model_kwargs={"cache_dir": self.cache_dir},
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map="auto",
            )

    def forward_call(self, prompt):
        sequences = self.pipeline(
            prompt,
            max_length=self.max_new_tokens,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]["generated_text"]

    def __call__(self, prompts):
        return [self.forward_call(prompt) for prompt in prompts]

class Mistral(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_dtype = torch.float16

        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights_path, 
            torch_dtype=self.torch_dtype, 
            device_map="auto",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts):
        final_prompts = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            final_prompts.append(prompt)
        
        _input = self.tokenizer(
            final_prompts,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            padding='max_length', 
            max_length=MAX_LENGTH).to(self.device)

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
    
    def truncate_response(self, response, max_length=500):
        tokens = self.tokenizer.tokenize(response)[:max_length]
        return self.tokenizer.convert_tokens_to_string(tokens)


class AutoModelCausalLMGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_dtype = torch.float16

        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights_path, 
            torch_dtype=self.torch_dtype, 
            device_map="auto",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def __call__(self, prompts, input_max_tokens=MAX_LENGTH):
        _input = self.tokenizer(
            prompts,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            max_length=input_max_tokens).to(self.device)

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
    
    def truncate_response(self, response, max_length=500):
        tokens = self.tokenizer.tokenize(response)[:max_length]
        return self.tokenizer.convert_tokens_to_string(tokens)
