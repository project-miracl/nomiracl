from nomiracl.generation import BaseGenerator

import requests
from transformers import AutoTokenizer
from typing import List
import os
import time

##########################
# AnyScale API Generator #
##########################
class AnyScale(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, cache_dir=self.cache_dir)
        
        self.api_base = os.getenv("ANYSCALE_BASE_URL")
        self.token = os.getenv("ANYSCALE_API_KEY")
        self.url = f"{self.api_base}/chat/completions"

        self.session = requests.Session()

    def __call__(self, prompts: List[str], n: int = 1) -> List[str]:
        responses = []
        
        for prompt in prompts:
            try:
                body = {
                    "model": self.weights_path,
                    "messages": [{"role": "user", "content": f"{prompt}"}],
                    "temperature": self.temperature
                }
                
                with self.session.post(
                    self.url, 
                    headers={"Authorization": f"Bearer {self.token}"}, 
                    json=body) as resp:
                    output = resp.json()["choices"][0]["message"]["content"]
                    responses.append(output)
            
            except Exception as e:
                print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
                time.sleep(self.wait)

                with self.session.post(
                    self.url, 
                    headers={"Authorization": f"Bearer {self.token}"}, 
                    json=body) as resp:
                    
                    try:
                        output = resp.json()["choices"][0]["message"]["content"]
                        responses.append(output)
                    
                    except Exception as e:  # noqa: E722
                        responses.append("Unable to generate response. reason: " + str(e))

        return responses
    
    def truncate_response(self, response: str, max_length: int = 500) -> str:
        tokens = self.tokenizer.tokenize(response)[:max_length]
        return self.tokenizer.convert_tokens_to_string(tokens)
