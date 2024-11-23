# Initial code structure taken from https://github.com/McGill-NLP/instruct-qa.

from .base import BaseGenerator
from typing import List

import torch
import logging
from transformers import pipeline
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BloomForCausalLM,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

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
        
        # LLAMA models are left-padded
        logger.info("Loading LLAMA model with left padding.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, padding_side="left"
        )

        ## exception for LLAMA-3
        if "llama-3" in self.weights_path.lower():
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        
        # check if the model is a PEFT model
        if self.peft:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.weights_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            self.model.eval()
        
        else:

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

        final_prompts = []

        # Apply chat template to the prompts
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            final_prompts.append(prompt)

        _input = self.tokenizer(
            final_prompts,
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
            eos_token_id=self.terminators if "llama-3" in self.weights_path.lower() else None
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
