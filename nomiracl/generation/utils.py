# Code modified from https://github.com/McGill-NLP/instruct-qa

from .huggingface import (
    Llama,
    Flan,
    Pipeline,
    Falcon,
    HFAutoModelCausalLM,
    BLOOM,
    Mistral,
)

from .cohere import Cohere
from .openai import OpenAIxNvidia
from .azure import GPTxAzure
from .anyscale import AnyScale
from .vllm import VLLM

RUNTYPES = {
    "huggingface": [Llama, Flan, Pipeline, Falcon, HFAutoModelCausalLM, BLOOM, Mistral],
    "cohere": [Cohere],
    "openai": [OpenAIxNvidia],
    "azure": [GPTxAzure],
    "anyscale": [AnyScale],
    "vllm": [VLLM],
}

def load_model(run_type: str, model_name: str, **kwargs):
    """
    Loads model by model_name available in Huggingface.

    Args:
        model_name (str): Name of model to load.
        kwargs: Additional parameters for the generator (e.g., temperature).

    Returns:
        BaseGenerator: Generator object.
    """
    run_type, model_name = run_type.lower(), model_name.lower()

    if run_type not in RUNTYPES:
        raise NotImplementedError(f"Run type {run_type} not supported. Choose from {list(RUNTYPES.keys())}.")
    
    if not any(model_type in model_name for model_type in RUNTYPES[run_type]):
        raise NotImplementedError(f"Model {model_name} not supported for run type {run_type}. Choose from {RUNTYPES[run_type]}.")
    
    model_cls = None

    ### Huggingface models
    if run_type == "huggingface":
        if "dolly" in model_name or "h2ogpt" in model_name: model_cls = Pipeline
        elif any(model_type in model_name for model_type in ["vicuna", "alpaca", "llama", "orca"]): model_cls = Llama
        elif any(model_type in model_name for model_type in ["mistral", "mixtral"]): model_cls = Mistral
        elif any(model_type in model_name for model_type in ["flan", "aya"]): model_cls = Flan
        elif any(model_type in model_name for model_type in ["zephyr", "gemma"]): model_cls = HFAutoModelCausalLM
        elif "falcon" in model_name: model_cls = Falcon
        elif "bloom" in model_name: model_cls = BLOOM
        else: raise NotImplementedError(f"Model {model_name} not supported.")
    
    ### Rest of the models
    else:
        model_cls = RUNTYPES[run_type][0]

    return model_cls(model_name, **kwargs)
