from nomiracl.generation import BaseGenerator

import requests
from typing import List
import os
import time
import logging
import tiktoken

logger = logging.getLogger(__name__)

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
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.model_name = self.model_name.replace("-azure", "")
        self.model_map = {
            "gpt-3.5-turbo": "chat",
            "gpt-4": "chat",
        }
        self.deployment_map = {
            "gpt-3.5-turbo": "gpt-35-turbo",
            "gpt-4": "gpt-4",
            "gpt-4o": "gpt-4o"
        }
        assert (
            self.model_name in self.model_map
        ), "You should add the model name to the model -> endpoint compatibility mappings."
        assert self.model_map[self.model_name] in [
            "chat",
        ], "Only chat endpoints are implemented. You may want to add other configurations."

        # json error happens if max_new_tokens is inf
        map_name = self.model_map[self.model_name]
        deployment_name = self.deployment_map[self.model_name]
        self.api_url = f"{AZURE_OPENAI_API_BASE}openai/deployments/{deployment_name}/{map_name}/completions?api-version={AZURE_OPENAI_API_VERSION}"
        self.max_new_tokens = self.max_new_tokens

    def __call__(self, prompts: List[str], n: int = 1) -> List[str]:
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