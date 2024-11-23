from nomiracl.generation import BaseGenerator

from typing import List
import os
import time
import tiktoken
import logging
from openai import OpenAI
from math import inf

from openai import (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    AuthenticationError,
    Timeout,
)

logger = logging.getLogger(__name__)

##################################
# OpenAI GPT-3.5/GPT-4 Generator #
##################################
class OpenAIxNvidia(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client = None
        # directly use the OpenAI API
        if os.getenv("OPENAI_API_KEY"):
            logger.info("Found OPENAI_API_KEY as an environment variable. Using OpenAI API.")
            logger.info(f"Loading {self.model_name} model using OpenAI API.")
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORGANIZATION"),
            )
        
        # use the NVIDIA API
        elif os.getenv("NVIDIA_API_KEY"):
            logger.info("Found NVIDIA_API_KEY as an environment variable. Using NVIDIA API.")
            logger.info(f"Loading {self.model_name} model using OpenAI API.")
            self.client = OpenAI(
                api_key=os.getenv("NVIDIA_API_KEY"),
                base_url = "https://integrate.api.nvidia.com/v1",
            )
        
        # json error happens if max_new_tokens is inf
        self.max_new_tokens = self.max_new_tokens

    def __call__(self, prompts: List[str], n: int = 1) -> List[str]:
        responses = []
        for prompt in prompts:
            kwargs = {
                "temperature": self.temperature, 
                "top_p": self.top_p, 
                "n": n
            }
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

    def api_request(self, prompt: str, **kwargs) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs)
            return [r.message.content for r in completion.choices]

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
    
    def get_tokens(self, response: str):
        encoder = tiktoken.encoding_for_model(self.model_name)
        return encoder.encode(response)
