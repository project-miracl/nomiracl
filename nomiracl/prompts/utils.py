# Code modified from https://github.com/McGill-NLP/instruct-qa.

from .template import VanillaTemplate


def load_prompt_template(template_name, **kwargs):
    """
    Loads prompts by template_name.

    Args:
        template_name (str): Name of prompt template to load.
        kwargs: Additional parameters for the template.

    Returns:
        PromptTemplate: Prompt template object.
    """
    if "vanilla" in template_name:
        template_cls = VanillaTemplate
    else:
        raise NotImplementedError(f"Template {template_name} not supported.")

    return template_cls(**kwargs)