# Code modified from https://github.com/McGill-NLP/instruct-qa

from .generator import (
    Llama,
    Flan,
    GPTx,
    Pipeline,
    Falcon,
    HFAutoModelCausalLM,
    GPTxAzure,
    BLOOM,
    Mistral,
)


def load_model(model_name, **kwargs):
    """
    Loads model by model_name available in Huggingface.

    Args:
        model_name (str): Name of model to load.
        kwargs: Additional parameters for the generator (e.g., temperature).

    Returns:
        BaseGenerator: Generator object.
    """
    model_name = model_name.lower()

    if "dolly" in model_name or "h2ogpt" in model_name:
        model_cls = Pipeline

    elif any(
        model_type in model_name for model_type in ["vicuna", "alpaca", "llama", "orca"]
    ):
        model_cls = Llama

    elif "azure" in model_name:
        model_cls = GPTxAzure

    elif "davinci" in model_name or "gpt" in model_name:
        model_cls = GPTx

    elif any(model_type in model_name.lower() for model_type in ["flan", "aya"]):
        model_cls = Flan

    elif "falcon" in model_name:
        model_cls = Falcon

    elif any(model_type in model_name.lower() for model_type in ["mistral", "mixtral"]):
        model_cls = Mistral

    elif any(model_type in model_name.lower() for model_type in ["zephyr", "gemma"]):
        model_cls = HFAutoModelCausalLM

    elif "bloom" in model_name.lower():
        model_cls = BLOOM

    else:
        raise NotImplementedError(f"Model {model_name} not supported.")

    return model_cls(model_name, **kwargs)
