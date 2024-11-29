from nomiracl.generation import BaseGenerator

import cohere
from transformers import AutoTokenizer
from typing import List
import os


########################
# Cohere API Generator #
########################
class Cohere(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load the Cohere tokenizer (sample one for now)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "CohereForAI/aya-23-35B", cache_dir=self.cache_dir
        )

        # load the Cohere API key
        self.cohere_key = os.getenv("COHERE_API_KEY")
        self.cohere = cohere.Client(self.cohere_key)
        self.deployment_map = {
            "command-r-plus": "command-r-plus",
            "aya-23": "c4ai-aya-23",
            "command-r": "command-r",
        }

    def __call__(self, prompts: List[str], n: int = 1) -> List[str]:

        # store the responses
        responses = []

        for prompt in prompts:
            try:
                deployment_name = self.deployment_map[self.model_name]
                # the number of return sequences is not supported by Cohere
                response = self.cohere.chat(
                    model=deployment_name,
                    message=prompt,
                    temperature=self.temperature,
                )
                responses.append(response.text)

            except Exception as e:
                print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
                responses.append("Unable to generate response.")

        return responses

    def truncate_response(self, response: str, max_length: int = 500) -> str:
        tokens = self.tokenizer.tokenize(response)[:max_length]
        return self.tokenizer.convert_tokens_to_string(tokens)
